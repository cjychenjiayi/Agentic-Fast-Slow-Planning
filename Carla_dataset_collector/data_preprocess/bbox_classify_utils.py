import numpy as np
import cv2
import time
def compute_color_prefix_sum(image, target_color):
    color_map = np.all(image == target_color, axis=-1).astype(np.int32)
    prefix_sum = np.cumsum(np.cumsum(color_map, axis=0), axis=1)
    return prefix_sum

def get_color_count_in_bbox(prefix_sums, bbox):
    x_min, y_min, x_max, y_max = bbox
    counts = []
    x_max = min(x_max, 1079)
    y_max = min(y_max, 719)
    for prefix_sum in prefix_sums:
        total = prefix_sum[y_max, x_max]
        if y_min > 0:
            total -= prefix_sum[y_min - 1, x_max]
        if x_min > 0:
            total -= prefix_sum[y_max, x_min - 1]
        if y_min > 0 and x_min > 0:
            total += prefix_sum[y_min - 1, x_min - 1]
        counts.append(total)
    return counts

def validate_and_calculate(image, bbox, target_colors, min_percentage=70, prefix_sums=None):
    x_min, y_min, x_max, y_max = bbox
    total_pixels = (x_max - x_min) * (y_max - y_min)
    color_counts = get_color_count_in_bbox(prefix_sums, bbox)
    percentages = [count / total_pixels * 100 for count in color_counts]
    max_percentage = max(percentages)
    
    best_category = percentages.index(max_percentage)
    sub_image = image[y_min:y_max, x_min:x_max]
    color_match = np.all(sub_image == target_colors[best_category], axis=-1)
    if color_match.any():
        sub_y_min, sub_x_min = np.min(np.where(color_match), axis=1)
        sub_y_max, sub_x_max = np.max(np.where(color_match), axis=1)
        y_min, x_min = sub_y_min + y_min, sub_x_min + x_min
        y_max, x_max = sub_y_max + y_min, sub_x_max + x_min
    else:
        return -1, max_percentage, None
    new_bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
    x_min, y_min, x_max, y_max = new_bbox
    total_pixels = (x_max - x_min) * (y_max - y_min)
    if total_pixels == 0:
        return -1, max_percentage, None
    new_count = get_color_count_in_bbox(prefix_sums, new_bbox)
    
    new_percentage = [count / total_pixels * 100 for count in new_count]
    new_max_percentage = max(new_percentage)
    if max_percentage < min_percentage and new_max_percentage < 35:
        return -1, max_percentage, None
    if max_percentage < min_percentage / 2:
        return -1, max_percentage, None
    return best_category, max_percentage, new_bbox


def optimize_bbox(image, bbox, target_colors, min_percentage=30, max_shift=20, prefix_sums=None):
    x_min, y_min, x_max, y_max = bbox
    bbox_width, bbox_height = x_max - x_min, y_max - y_min
    
    best_category, best_percentage, new_bbox = validate_and_calculate(image, bbox, target_colors, min_percentage, prefix_sums)

    if best_category == -1:
        return best_category, bbox
    
    for dx in range(-max_shift, max_shift + 1, 3):
        for dy in range(-max_shift, max_shift + 1, 3):
            new_x_min = max(0, min(image.shape[1] - bbox_width - 1, x_min + dx))
            new_y_min = max(0, min(image.shape[0] - bbox_height - 1, y_min + dy))
            new_x_max, new_y_max = new_x_min + bbox_width, new_y_min + bbox_height
            category, percentage, _ = validate_and_calculate(image, (new_x_min, new_y_min, new_x_max, new_y_max), target_colors, min_percentage, prefix_sums)
            if category != -1 and percentage >= best_percentage:
                best_category, best_percentage, new_bbox = category, percentage, _
    
    return best_category, new_bbox

def resize_bboxes(image_path, bboxes, categories, min_percentage=25, max_shift=10):
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        return [], []
    # please ref to the segmentation mask on tutorial of CARLA SIMULATOR
    cat_list = [[(0, 0, 142)], [(170, 120, 50), (110, 190, 160)]]
    prefix_sums_map = {}
    for colors in cat_list:
        for color in colors:
            prefix_sums_map[color] = np.array(compute_color_prefix_sum(image, color))
    results = []
    new_bbox = []
    for bbox, cat in zip(bboxes, categories):
        if cat == -1:
            results.append(-1)
            new_bbox.append(None)
            continue
        target_colors = cat_list[cat]
        prefix_sums = [prefix_sums_map[color] for color in target_colors]
        category, bbox = optimize_bbox(image, bbox, target_colors, min_percentage, max_shift, prefix_sums)
        results.append(category)
        new_bbox.append(bbox)
    return results, new_bbox

