import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from nets.siamese import CMCNet
from utils.utils import cvtColor
# --- 辅助函数 (已修正 yolo_to_norm_corners) ---

def load_boxes_from_file(filepath):
    boxes = []
    if not os.path.exists(filepath):
        return boxes
    with open(filepath, 'r') as f:
        for line in f.readlines():
            parts = [float(p) for p in line.strip().split()]
            if len(parts) >= 5:
                # 返回格式: [[class_id, xc, yc, w, h], ...]
                boxes.append(parts)
    return boxes

def yolo_to_norm_corners(box):
    """
    修正: 直接解包传入的 4 个坐标 [xc, yc, w, h]。
    外部调用者 (evaluate_results) 已经完成了正确的切片。
    """
    if box is None or len(box) < 4:
        # 安全检查
        return [0.0, 0.0, 0.0, 0.0]

    # 修正的核心: 移除不必要的切片 [1:5]
    x_center, y_center, w, h = box 
    
    x1 = max(0, x_center - w / 2)
    y1 = max(0, y_center - h / 2)
    x2 = min(1, x_center + w / 2)
    y2 = min(1, y_center + h / 2)
    return [x1, y1, x2, y2]

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    unionArea = boxAArea + boxBArea - interArea + 1e-6
    return interArea / unionArea

# --- 配置 (保留不变) ---
BEST_MODEL_PATH = '/kaggle/working/logs/ep020-loss1.689-val1.724.pth'
TEST_PATCH_DIR = '/kaggle/input/siamesedata/siameseD/test'
GT_LABEL_DIR = '/kaggle/input/breast/data'
YOLO_PRED_LABEL_DIR = '/kaggle/input/siamesedata/siameseD'
INPUT_SHAPE = [224, 224]
NUM_CLASSES = 4
CONFIDENCE_THRESHOLD = 0.0001

# --- Full Detection Pipeline (保留不变) ---
class FullDetectionPipeline:
    def __init__(self, model, gt_base_dir, patch_base_dir, yolo_pred_dir):
        self.net = model.eval()
        self.gt_base_dir = gt_base_dir
        self.patch_base_dir = patch_base_dir
        self.yolo_pred_dir = yolo_pred_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_shape = INPUT_SHAPE
        
        self.cc_proposals = self._get_all_proposals('cc')
        self.mlo_proposals = self._get_all_proposals('mlo')
        self.all_test_pairs = self._generate_cross_view_pairs()
        print(f"Total {len(self.all_test_pairs)} cross-view proposal pairs generated.")
        
        self.yolo_box_map = self._load_all_yolo_boxes()

    def _get_all_proposals(self, view):
        proposals = {}
        path = os.path.join(self.patch_base_dir, view, 'crops', '*.jpg')
        for file_path in glob.glob(path):
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            if len(parts) >= 3:
                match_key = f"{parts[0]}_{parts[1]}"
                if match_key not in proposals:
                    proposals[match_key] = []
                proposals[match_key].append({'path': file_path, 'filename': filename})
        return proposals

    def _load_all_yolo_boxes(self):
        yolo_box_map = {}
        for view in ['cc', 'mlo']:
            pred_dir = os.path.join(self.yolo_pred_dir, 'test', view, 'labels')
            for file_path in glob.glob(os.path.join(pred_dir, '*_*.txt')):
                filename = os.path.basename(file_path)
                name_without_ext = os.path.splitext(filename)[0]
                boxes = load_boxes_from_file(file_path)
                # 假设每个 YOLO 预测文件只包含一个主要检测框
                if boxes: # 这里的 boxes[0] 是 [class_id, xc, yc, w, h] 列表 
                    yolo_box_map[name_without_ext] = boxes[0][1:5] # 提取 [xc, yc, w, h]
        return yolo_box_map

    def _generate_cross_view_pairs(self):
        all_pairs = []
        for match_key in self.cc_proposals.keys():
            if match_key in self.mlo_proposals:
                for cc_prop in self.cc_proposals[match_key]:
                    for mlo_prop in self.mlo_proposals[match_key]:
                        all_pairs.append({'key': match_key, 'cc': cc_prop, 'mlo': mlo_prop})
        return all_pairs

    def _load_and_preprocess(self, path):
        image = Image.open(path).convert('RGB')
        image = image.resize((self.input_shape[1], self.input_shape[0]))
        image = np.array(image, dtype=np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        return tensor

    def run_inference(self):
        results = {}
        for pair_data in tqdm(self.all_test_pairs, desc="CMCNet Inference"):
            img_cc = self._load_and_preprocess(pair_data['cc']['path'])
            img_mlo = self._load_and_preprocess(pair_data['mlo']['path'])
            
            with torch.no_grad():
                out_cls_cc, out_cls_mlo, out_match, _, _ = self.net(img_cc, img_mlo)
                
                # 获取 cc 侧的分类概率和置信度 (作为最终输出的依据)
                prob_cls_cc = F.softmax(out_cls_cc, dim=1).cpu().numpy().squeeze()
                pred_cls_cc = np.argmax(prob_cls_cc)
                cc_confidence = prob_cls_cc[pred_cls_cc]

                prob_cls_mlo = F.softmax(out_cls_mlo, dim=1).cpu().numpy().squeeze()
                pred_cls_mlo = np.argmax(prob_cls_mlo)
                
                prob_match = F.softmax(out_match, dim=1).cpu().numpy().squeeze()
                matching_score = prob_match[1]

            match_key = pair_data['key']
            
            # 使用 cc 侧的分类置信度作为最终 Mass 预测的置信度 (mAP所需)
            is_mass_cc = (pred_cls_cc < 3) and (cc_confidence >= CONFIDENCE_THRESHOLD)
            is_mass_mlo = (pred_cls_mlo < 3) # mlo 侧只需要是 Mass 类
            is_eligible = is_mass_cc and is_mass_mlo

            if match_key not in results:
                results[match_key] = {'best_score': -1.0, 'final_cc_proposal': None, 
                                      'predicted_class': 3, 'predicted_box_coords': None,
                                      'confidence_score': 0.0} # 默认背景类

            # 只有当它是合格的 Mass 预测 (is_eligible) 且匹配得分更高时，才更新
            if is_eligible and matching_score > results[match_key]['best_score']:
                cc_pred_name = os.path.splitext(pair_data['cc']['filename'])[0]
                predicted_box_coords = self.yolo_box_map.get(cc_pred_name, None)
                
                # 只有当找到对应的 YOLO 框时才更新
                if predicted_box_coords is not None:
                    results[match_key]['best_score'] = matching_score
                    results[match_key]['final_cc_proposal'] = pair_data['cc']
                    results[match_key]['predicted_box_coords'] = predicted_box_coords
                    results[match_key]['predicted_class'] = pred_cls_cc
                    # 使用 cc 侧的分类置信度作为 mAP 所需的最终得分
                    results[match_key]['confidence_score'] = cc_confidence 
        
        # 处理未成功预测 Mass 的样本，确保其 'predicted_class' = 3, 'confidence_score' = 0
        for key in results:
            if results[key]['final_cc_proposal'] is None:
                results[key]['predicted_class'] = 3
                results[key]['confidence_score'] = 0.0

        return results

# --- mAP & AP 辅助函数 (保持不变) ---
def calculate_ap(tp_sum, fp_sum, num_gt):
    if num_gt == 0:
        return 0.0
    precision = np.divide(tp_sum, (fp_sum + tp_sum))
    recall = np.divide(tp_sum, num_gt)
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
    return ap

def calculate_map(all_preds_for_map, all_gt_for_map, iou_thresholds):
    all_ap = []
    predictions = []
    for img_name, preds in all_preds_for_map.items():
        for pred in preds:
            predictions.append({
                'img': img_name, 'box': pred['box'], 'score': pred['score'], 'class': pred['class']
            })
    predictions.sort(key=lambda x: x['score'], reverse=True)
    num_gt_total = sum(len(gt_list) for gt_list in all_gt_for_map.values())
    if num_gt_total == 0:
        return 0.0

    for iou_thresh in iou_thresholds:
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        gt_matched = {name: [False] * len(gts) for name, gts in all_gt_for_map.items()}

        for i, pred in enumerate(predictions):
            img_name = pred['img']
            pred_box = yolo_to_norm_corners(pred['box'])
            best_iou = 0.0
            best_gt_idx = -1
            
            if img_name in all_gt_for_map:
                for j, gt_data_dict in enumerate(all_gt_for_map[img_name]):
                    gt_box = yolo_to_norm_corners(gt_data_dict['box']) 
                    current_iou = calculate_iou(pred_box, gt_box)
                    
                    if current_iou > best_iou:
                        best_iou = current_iou
                        best_gt_idx = j

            if best_iou >= iou_thresh and best_gt_idx != -1 and not gt_matched[img_name][best_gt_idx]:
                tp[i] = 1.0
                gt_matched[img_name][best_gt_idx] = True
            else:
                fp[i] = 1.0

        tp_sum = np.cumsum(tp)
        fp_sum = np.cumsum(fp)
        ap = calculate_ap(tp_sum, fp_sum, num_gt_total)
        all_ap.append(ap)

    mean_average_precision = np.mean(all_ap)
    return mean_average_precision


# --- 最终评估函数 (已修正) ---
def evaluate_results(final_predictions, gt_base_dir, iou_threshold=0.0001):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0 
    
    # --- mAP 数据收集部分 ---
    all_gt_for_map = {} 
    all_preds_for_map = {} 

    for match_key, prediction_data in final_predictions.items():
        predicted_box_coords = prediction_data.get('predicted_box_coords')
        predicted_class_id = prediction_data['predicted_class']
        is_mass_prediction = predicted_class_id < 3

        cc_prop_info = prediction_data['final_cc_proposal']
        confidence_score = prediction_data.get('confidence_score', 0.0) 

        # --- 1. GT 检查 ---
        if cc_prop_info is None:
            # 假设 match_key 格式为 base_name_CC/MLO，取 base_name
            parts = match_key.split('_')
            original_base_name = "_".join(parts[:-1]) if parts[-1] in ('CC', 'MLO') else match_key 
        else:
            original_base_name = "_".join(cc_prop_info['filename'].split('_')[:3])
            
        cc_gt_path = os.path.join(gt_base_dir, 'cc_view', 'labels', 'test', original_base_name + '.txt')
        gt_boxes_data = load_boxes_from_file(cc_gt_path)
        has_ground_truth = len(gt_boxes_data) > 0
        
        # 收集 GT 数据 (将列表格式转换为字典格式供 mAP 函数使用)
        if original_base_name not in all_gt_for_map:
            converted_gt_list = []
            for gt_data_list in gt_boxes_data:
                if len(gt_data_list) >= 5:
                    converted_gt_list.append({
                        'box': gt_data_list[1:5],      # [xc, yc, w, h]
                        'class': int(gt_data_list[0])  # class_id
                    })
            all_gt_for_map[original_base_name] = converted_gt_list
             
        # 收集预测数据
        if is_mass_prediction and predicted_box_coords is not None:
            if original_base_name not in all_preds_for_map:
                all_preds_for_map[original_base_name] = []
                
            all_preds_for_map[original_base_name].append({
                'box': predicted_box_coords,
                'score': confidence_score,
                'class': predicted_class_id 
            })

        # --- 2. 预测与统计 (TP/FP/FN/TN 逻辑) ---
        is_true_positive = False
        
        # A. 如果有成功预测，检查 IoU (TP/FP)
        if cc_prop_info is not None and predicted_box_coords is not None:
            
            if is_mass_prediction and has_ground_truth:
                pred_box_norm = yolo_to_norm_corners(predicted_box_coords)
                
                # 修正: 使用索引访问 load_boxes_from_file 返回的列表数据
                for gt_data_list in gt_boxes_data:
                    if len(gt_data_list) >= 5:
                        gt_coords = gt_data_list[1:5] # 提取 [xc, yc, w, h] (4个元素)
                        gt_box_norm = yolo_to_norm_corners(gt_coords)
                        
                        if calculate_iou(pred_box_norm, gt_box_norm) >= iou_threshold:
                            is_true_positive = True
                            break
            
            # 统计 TP 和 FP
            if is_mass_prediction and is_true_positive:
                total_tp += 1
            elif is_mass_prediction and not is_true_positive:
                total_fp += 1
        
        # B. 统计 FN 和 TN
        if has_ground_truth and not is_true_positive:
             total_fn += len(gt_boxes_data) 
            
        if not is_mass_prediction and not has_ground_truth:
             total_tn += 1


    # --- 最终指标计算 ---
    total_detections = total_tp + total_fp
    total_ground_truth = total_tp + total_fn
    total_samples = total_tp + total_fp + total_fn + total_tn

    precision = total_tp / total_detections if total_detections > 0 else 0
    recall = total_tp / total_ground_truth if total_ground_truth > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (total_tp + total_tn) / total_samples if total_samples > 0 else 0

    # --- 计算 mAP ---
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    map_05_095 = calculate_map(all_preds_for_map, all_gt_for_map, iou_thresholds) 
    
    return {
        'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'TN': total_tn,
        'Total_Samples': total_samples,
        'Precision': precision, 'Recall': recall, 'F1_Score': f1_score,
        'Accuracy': accuracy,
        'mAP_0.5:0.95': map_05_095
    }

# --- 主函数 (保留不变) ---
def main_test_pipeline(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 假设 CMCNet 是在其他地方定义的
    model = CMCNet(num_classes=NUM_CLASSES, pretrained=False) 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    pipeline = FullDetectionPipeline(model, GT_LABEL_DIR, TEST_PATCH_DIR, YOLO_PRED_LABEL_DIR)
    final_predictions = pipeline.run_inference()
    results = evaluate_results(final_predictions, GT_LABEL_DIR)

    print("\n--- 测试结果 ---")
    for k,v in results.items():
        print(f"{k}: {v}")

if __name__ == '__main__':
    # 假设 CMCNet 和 utils.utils.cvtColor 已经被正确导入
    main_test_pipeline(BEST_MODEL_PATH)
