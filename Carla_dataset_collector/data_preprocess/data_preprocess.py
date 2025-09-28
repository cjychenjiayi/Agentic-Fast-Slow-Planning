import json
import numpy as np
import cv2
import os
from bbox_classify_utils import resize_bboxes
from check_intersection import check_iou, check_inside
import math 
import tqdm
import concurrent.futures
import random
import time
random.seed(time.time())

def check_for_invalid_values(data):
    if isinstance(data, dict):
        for value in data.values():
            if check_for_invalid_values(value) is None:
                return None
    elif isinstance(data, list):
        for item in data:
            if check_for_invalid_values(item) is None:
                return None
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
    return data

def draw_bounding_boxes(image, bbox_list, color=(0, 0, 255), thickness=2):
    for bbox in bbox_list:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image
def cal_dis(loc_1, loc_2):
    x1, y1, z1 = loc_1["x"], loc_1["y"], loc_1["z"]
    x2, y2, z2 = loc_2["x"], loc_2["y"], loc_2["z"]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance
def process_json_and_validate(bbox_data, image_path, seg_image_path, vehicle_loc, output_image_path=None):
    try:
        image = cv2.imread(image_path)
        height, width, _ = image.shape
    except:
        print("AAAAAAAAAAA")
        exit(1)
        return[]
    validations = []
    real_bbox = []
    category = []
    real_category = []
    dist_and_orient = []
    flag = False
    for obj in bbox_data:
        bbox = obj['bbox']
        cat = 0 if str(obj["id"]).startswith("vehicle") else 1
        loc = obj["location"]
        obj_orient = obj["related_orient"] if "related_orient" in obj else None
        x_min, y_min, x_max, y_max = map(int, [bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']])
        bbox = [x_min, y_min, x_max, y_max]
        is_available = check_inside(bbox, width, height)
        if is_available == 0:
            continue
        if is_available == -1:
            flag = True
            break
        bbox = [max(0, x_min), max(0, y_min), min(width-1, x_max), min(height-1, y_max)]
        category.append(cat)
        real_category.append(obj["id"])
        real_bbox.append(bbox)
        dist_and_orient.append([cal_dis(loc, vehicle_loc), obj_orient])
    if flag:
        return []
    category, real_bbox = resize_bboxes(seg_image_path, real_bbox, category)
    for bbox, cat, real_cat, distandorient in zip(real_bbox, category, real_category, dist_and_orient):
        if cat != -1:
            validations.append(
                {
                    "bbox": bbox,
                    "category": cat,
                    "real_cat": real_cat,
                    "dist": distandorient[0],
                    "orient": distandorient[1]
                }
            )
    validations = check_iou(validations)
    if output_image_path is not None:
        image = cv2.imread(image_path)
        image_with_bboxes = draw_bounding_boxes(image, [obj['bbox'] for obj in validations], color=(0, 255, 0), thickness=3)
        cv2.imwrite(output_image_path, image_with_bboxes)
    return validations

def find_path(json_path):
    image_path = json_path.replace("json", "png").replace("/data/", "/rgb/")
    seg_image_path = json_path.replace("json", "png").replace("/data/", "/seg/")
    return image_path, seg_image_path

def select_path(folder_path):
    result = {}
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            json_dict = {}
            for sub_subfolder in os.listdir(subfolder_path):
                sub_subfolder = os.path.join(subfolder_path, sub_subfolder)
                if os.path.isdir(sub_subfolder) == False:
                    continue
                json_folder = os.path.join(sub_subfolder, "data")
                json_dict[sub_subfolder] = []
                for file_name in os.listdir(json_folder):
                    if file_name.endswith('.json'):
                        json_dict[sub_subfolder].append(os.path.join(json_folder, file_name))
            tot_amount = sum([len(value) for key, value in json_dict.items()])
            if tot_amount > 0:
                result[subfolder_name] = json_dict
    return result

def check_loc(loc_1, loc_2):
    x1, y1, z1 = loc_1["x"], loc_1["y"], loc_1["z"]
    x2, y2, z2 = loc_2["x"], loc_2["y"], loc_2["z"]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance > 2.0

def select_list_json(json_list, desc_name):
    lst = {"x": -10000, "y": -10000, "z": -10000}
    ret = []
    lst_loc = None
    # try:
    for json_path in tqdm.tqdm(json_list, desc=desc_name):
        with open(json_path, 'r') as file:
            json_data = json.load(file)
        json_data = check_for_invalid_values(json_data)
        if json_data is None:
            continue
        loc = json_data[0]["vehicle_location"]
        bbox_data = json_data[1]
        if lst_loc is not None and check_loc(lst_loc, loc) == False:
            continue
        image_path, seg_image_path = find_path(json_path)
        validations = process_json_and_validate(bbox_data, image_path, seg_image_path, loc)
        if len(validations) <= 0:
            continue
        yaw = json_data[0]["vehicle_rotation"] if "vehicle_rotation" in json_data[0] else None
        ret.append({"path": json_path, "loc": loc, "bbox": validations, "yaw":yaw})
        lst_loc = loc
    # except:
    #     return []
    return ret

def multithread(trial_name, single_trial, json_list, pbar):
    result = select_list_json(json_list, f"{trial_name}_{single_trial}")
    pbar.update(1)
    return trial_name, single_trial, result

def process_trial(json_path_dict):
    os.makedirs("result1and2", exist_ok=True)
    task_lists = []
    for trial_name, trial_data in json_path_dict.items():
        for single_trial, json_list in trial_data.items():
            single_trial = single_trial.split("/")[-1]
            task_lists.append([trial_name, single_trial, json_list])
    result_dict = {}
    random.shuffle(task_lists)
    print(len(task_lists))
    with tqdm.tqdm(total=len(task_lists)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
            future_list = [
                executor.submit(multithread, trial_name, single_trial, json_list, pbar)
                for trial_name, single_trial, json_list in task_lists
            ]
            for future in concurrent.futures.as_completed(future_list):
                try:
                    trial_name, single_trial, result = future.result()
                    path = f"result1and2/{trial_name}_{single_trial}.json"
                    if len(result) == 0:
                        print(f"EMPTY in {path}")
                        continue
                    if trial_name not in result_dict:
                        result_dict[trial_name] = {}
                    result_dict[trial_name][single_trial] = result
                    
                    with open(path, "w") as f:
                        json.dump(result, f, indent=4)
                    print(f"FINISH saving {path}")
                except Exception as exc:
                    print(f"Trial {single_trial} generated an exception: {exc}")
                    exit(1)
    
    return result_dict

if __name__ == "__main__":

    folder_path_list = [
                    "YOUR_FOLDERS"
                  ]
    tot_json_dict = {}
    cnt = 0
    for folder_path in folder_path_list:
        json_path_dict = select_path(folder_path)
        for trial_name, trial_data in json_path_dict.items():
            if trial_name not in tot_json_dict:
                tot_json_dict[trial_name] = {}
            tot_json_dict[trial_name].update(trial_data)
    result_dict = process_trial(tot_json_dict)
    with open("result.json", "w") as f:
        json.dump(result_dict, f, indent=4)
