from parse_result_utils import *
from judgement import *
from batch_inference import eval_model
def get_format(data):
    return f"{data['ref']} {data['dist']}m {data['orient']} deg"
if __name__ == "__main__":
    jsonl_file_path = "YOUR_PATH"
    replace_folder = None
    data = parse_jsonl(jsonl_file_path, replace_folder)

    image_paths = [ d["image"] for d in data ]
    prompt = "Please detect all the object in this image with corresponding bounding box, distance and orientation"
    questions = [prompt] * len(image_paths)
    all_output_list = eval_model(questions, image_paths)
    
    all_output_list = [parse_data_internvl(output, image) for output, image in zip(all_output_list, image_paths)]
    gt_ans = [ d["gpt_value"] for d in data]
    tot_dict = {}

    key_list = {"unmatched_preds":0, "unmatched_gts":0}
    for output, gt in zip(all_output_list, gt_ans):
        result = judge(gt, output)
        for key, value in result.items():
            if key in key_list:
                key_list[key] += value
                continue
            if key not in tot_dict:
                tot_dict[key] = []
            tot_dict[key].extend(value)
    for key, value in key_list.items():
        print(key, value / len(gt_ans))
    for key, value in tot_dict.items():
        average_value = sum(value) / len(value)
        print(key, average_value)
    with open("middle_final.json", "w") as f:
        for key, value in tot_dict.items():
            tot_dict[key] = sum(value) / len(value) 
        json.dump(tot_dict, f, indent=4)
    ret = []
    for image_path, gt, pred in zip(image_paths, gt_ans, all_output_list):
        ret.append({"image":image_path, "gt":gt, "pred":pred})
    with open("result_final.json", "w") as f:
        json.dump(ret, f, indent=4)
    
    for image_path, pred in zip(image_paths, all_output_list):
        bbox_dict = [[data["box"],get_format(data)] for data in pred]
        visualize(bbox_dict, image_path)