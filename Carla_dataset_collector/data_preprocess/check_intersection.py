def calculate_intersection_area(bbox1, bbox2):
    xmin_inter = max(bbox1[0], bbox2[0])
    ymin_inter = max(bbox1[1], bbox2[1])
    xmax_inter = min(bbox1[2], bbox2[2])
    ymax_inter = min(bbox1[3], bbox2[3])
    if xmin_inter >= xmax_inter or ymin_inter >= ymax_inter:
        return 0
    return (xmax_inter - xmin_inter) * (ymax_inter - ymin_inter)

def check_iou(validations):
    n = len(validations)
    to_remove = set() 
    areas = []
    for i in range(n):
        bbox = validations[i]["bbox"]
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        areas.append(area)
    
    for i in range(n):
        for j in range(i + 1, n):
            bbox1 = validations[i]["bbox"]
            bbox2 = validations[j]["bbox"]
            intersection_area = calculate_intersection_area(bbox1, bbox2)
            if intersection_area == 0:
                continue
            area1 = areas[i]
            area2 = areas[j]
            overlap1 = intersection_area / area1
            overlap2 = intersection_area / area2
            if overlap1 >= 0.8:
                to_remove.add(i)
            elif overlap2 >= 0.8:
                to_remove.add(j)
            if 0.6 <= overlap1 < 0.8 or 0.6 <= overlap2 < 0.8:
                return [] 

    return [validation for idx, validation in enumerate(validations) if idx not in to_remove]

def check_inside(bbox, width, height, upper_threshold=0.7):
    image_area = width * height
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    bbox_image = [0, 0, width-1, height-1]
    intersection_area = calculate_intersection_area(bbox, bbox_image)
    intersection_ratio = intersection_area / bbox_area if bbox_area > 0 else 0
    if intersection_ratio >= upper_threshold and intersection_area / image_area > 0.0002 and intersection_area / image_area < 0.1:
        return 1
    intersection_ratio_image = intersection_area / image_area
    if intersection_ratio_image > 0.1:
        return 1
    elif intersection_ratio_image >= 0.03:
        return -1
    else:
        return 0

    