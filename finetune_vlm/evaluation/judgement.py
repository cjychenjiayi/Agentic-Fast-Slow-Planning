import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_iou(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    if x_inter_min >= x_inter_max or y_inter_min >= y_inter_max:
        return 0.0
    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    return inter_area / (bbox1_area + bbox2_area - inter_area)

def compute_difference(pred, gt):
    category_diff = 0 if pred['ref'] == gt['ref'] else 1
    iou = compute_iou(pred['box'], gt['box'])
    dist_diff = abs(pred['dist'] - gt['dist'])
    dist_diff_percent = dist_diff / gt["dist"]
    orient_diff = min(abs(pred['orient'] - gt['orient']), 360 - abs(pred['orient'] - gt['orient']))
    # orient_diff_percent = orient_diff / abs(gt["orient"])
    return {'category_diff': category_diff, 'iou': iou, 'dist_diff': dist_diff, 'orient_diff': orient_diff, "dist_diff_percent": dist_diff_percent}

def match_predictions(ground_truths, predictions, iou_threshold=0.5):
    iou_matrix = np.zeros((len(predictions), len(ground_truths)))

    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truths):
            iou_matrix[i, j] = compute_iou(pred['box'], gt['box'])
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matched_pairs = []
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= iou_threshold:
            matched_pairs.append({'pred': predictions[i], 'gt': ground_truths[j], 'iou': iou_matrix[i, j]})
    
    return matched_pairs

def evaluate_match(matched_pairs, predictions, ground_truths):
    total_category_diff = total_iou = total_dist_diff = total_orient_diff = 0
    total_percent_dis = total_percent_orient = 0
    total_pairs = len(matched_pairs)
    unmatched_preds = len(predictions) - total_pairs
    unmatched_gts = len(ground_truths) - total_pairs

    for pair in matched_pairs:
        diff = compute_difference(pair['pred'], pair['gt'])
        total_category_diff += diff['category_diff']
        total_iou += diff['iou']
        total_dist_diff += diff['dist_diff']
        total_orient_diff += diff['orient_diff']
        total_percent_dis += diff['dist_diff_percent']
        # total_percent_orient += diff['orient_diff_percent']

    avg_category_diff = total_category_diff / total_pairs if total_pairs else 0
    avg_iou = total_iou / total_pairs if total_pairs else 0
    avg_dist_diff = total_dist_diff / total_pairs if total_pairs else 0
    avg_orient_diff = total_orient_diff / total_pairs if total_pairs else 0
    avg_dis_percent = total_percent_dis / total_pairs if total_pairs else 0
    # avg_orient_percent = total_percent_orient / total_pairs if total_pairs else 0
    return {
        'avg_category_acc': avg_category_diff,
        'avg_iou': avg_iou,
        'avg_dist_diff': avg_dist_diff,
        'avg_orient_diff': avg_orient_diff,
        'unmatched_preds': unmatched_preds,
        'unmatched_gts': unmatched_gts,
        "avg_dist_percent": avg_dis_percent,
        # "avg_orient_percent": avg_orient_percent
    }

def evaluate_match_dict(matched_pairs, predictions, ground_truths):
    category_diffs = []
    iou_vals = []
    dist_diffs = []
    orient_diffs = []
    percent_dis = []
    percent_orient = []
    
    total_pairs = len(matched_pairs)
    unmatched_preds = len(predictions) - total_pairs
    unmatched_gts = len(ground_truths) - total_pairs

    for pair in matched_pairs:
        diff = compute_difference(pair['pred'], pair['gt'])
        category_diffs.append(diff['category_diff'])
        iou_vals.append(diff['iou'])
        dist_diffs.append(diff['dist_diff'])
        orient_diffs.append(diff['orient_diff'])
        percent_dis.append(diff.get('dist_diff_percent', 0))
        percent_orient.append(diff.get('orient_diff_percent', 0))

    return {
        'category_diffs': category_diffs,
        'iou_vals': iou_vals,
        'dist_diffs': dist_diffs,
        'orient_diffs': orient_diffs,
        'unmatched_preds': unmatched_preds,
        'unmatched_gts': unmatched_gts,
        'percent_dis': percent_dis,
        'percent_orient': percent_orient
    }

def judge(ground_truths, predictions, list_only = True):
    matched_pairs = match_predictions(ground_truths, predictions)
    if list_only:
        return evaluate_match_dict(matched_pairs, predictions, ground_truths)
    return evaluate_match(matched_pairs, predictions, ground_truths)

if __name__ == "__main__":
    ground_truths = [{'ref': 'vehicle', 'box': [542, 572, 760, 855], 'dist': 6.0, 'orient': -15},
                    {'ref': 'table', 'box': [0, 590, 282, 766], 'dist': 5.9, 'orient': 35}]
    predictions = [{'ref': 'vehicle', 'box': [540, 570, 758, 850], 'dist': 6.1, 'orient': -14},
                {'ref': 'table', 'box': [5, 590, 285, 765], 'dist': 6.0, 'orient': 34},
                {'ref': 'vehicle', 'box': [600, 600, 700, 700], 'dist': 7.0, 'orient': 0}]

    evaluation_metrics = judge(ground_truths, predictions)
    print(evaluation_metrics)
