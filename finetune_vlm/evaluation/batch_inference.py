import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
import torch
from torchvision.transforms.functional import InterpolationMode
import tqdm
import json
import multiprocessing
from parse_result_utils import *
import warnings
import shutil
warnings.filterwarnings("ignore")


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def batch_load_images(batch_image_paths, max_num=6):
    pixel_values_list = []
    num_patches_list = []

    for image_path in batch_image_paths:
        pixel_values = load_image(image_path, max_num=max_num).to(torch.bfloat16).cuda()
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.size(0))
    
    all_pixel_values = torch.cat(pixel_values_list, dim=0)
    return all_pixel_values, num_patches_list

copy_file_path = "YOUR_DOWNLOAD_MODEL"
model_path = "YOUR_FINETUNED_MODEL"

def copy_py_files(src_folder, dst_folder): 
    # due to someof the version mismatch, it's better to copy the config python file from origin
    os.makedirs(dst_folder, exist_ok=True)
    for file_name in os.listdir(src_folder):
        if file_name.endswith(".py"):
            src_file = os.path.join(src_folder, file_name)
            dst_file = os.path.join(dst_folder, file_name)
            shutil.copy2(src_file, dst_file)
copy_py_files(copy_file_path, model_path)


kwargs = {'device_map': 'cuda'} 
generation_config = dict(
    max_new_tokens=4096,
    do_sample=True,
    temperature = 0.7
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

def process_on_gpu(gpu_id, task_batches, max_num=12, length = 0):
    torch.cuda.set_device(gpu_id)
    local_model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    ).eval().cuda() 
    
    if local_model.config.pad_token_id is None:
        local_model.config.pad_token_id = tokenizer.eos_token_id
    local_output_list = [] 

    batch_questions_list, batch_image_paths_list = task_batches[0], task_batches[1]
    for batch_questions, batch_image_paths in tqdm.tqdm(zip(batch_questions_list, batch_image_paths_list), desc=f"GPUS: {gpu_id}", total = length):
        # print(batch_questions, batch_image_paths)
        pixel_values, num_patches_list = batch_load_images(batch_image_paths, max_num=max_num)
        responses = local_model.batch_chat(
            tokenizer, 
            pixel_values, 
            num_patches_list=num_patches_list, 
            questions=batch_questions, 
            generation_config=generation_config
        )
        local_output_list.extend(responses)
        
    return local_output_list

def eval_model(questions, image_paths, batch_size=4, max_num=12, num_gpus=4):
    batches_questions = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]
    batches_image_paths = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
    n = len(batches_questions)
    base_size = n // num_gpus
    remainder = n % num_gpus
    gpu_batches = []
    start = 0
    for i in range(num_gpus):
        cur_index = base_size + (1 if i < remainder else 0)
        gpu_batches.append([batches_questions[start:start + cur_index], batches_image_paths[start:start + cur_index]])
        start += cur_index
    with multiprocessing.Pool(processes=num_gpus) as pool:
        results = [
            pool.apply_async(process_on_gpu, args=(gpu_id, gpu_batches[gpu_id], max_num, len(list(gpu_batches[gpu_id][0])) ))
            for gpu_id in range(num_gpus)
        ]
        all_output_list = []
        for result in results:
            all_output_list.extend(result.get())
    return all_output_list

if __name__ == "__main__":
    pass