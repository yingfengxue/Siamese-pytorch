import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# 安装库（只需执行一次）
# !pip install mean-average-precision

from mean_average_precision import MetricBuilder

# --- 导入模型 ---
from nets.siamese import CMCNet
from utils.utils import cvtColor

# --- 辅助函数 ---
def load_boxes_from_file(filepath):
    boxes = []
    if not os.path.exists(filepath):
        return boxes
    with open(filepath, 'r') as f:
        for line in f.readlines():
            parts = [float(p) for p in line.strip().split()]
            if len(parts) >= 5:
                boxes.append(parts)
    return boxes

def yolo_to_norm_corners(box):
    x_center, y_center, w, h = box[1:5]
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

# --- 配置 ---
BEST_MODEL_PATH = '/kaggle/working/logs/ep020-loss1.689-val1.724.pth'
TEST_PATCH_DIR = '/kaggle/input/siamesedata/siameseD/val'
GT_LABEL_DIR = '/kaggle/input/breast/data'
YOLO_PRED_LABEL_DIR = '/kaggle/input/siamesedata/siameseD'
INPUT_SHAPE = [224, 224]
NUM_CLASSES = 4
CONFIDENCE_THRESHOLD = 0.01

# ---------------------------
# Full Detection Pipeline
# ---------------------------
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
                if boxes:
                    yolo_box_map[name_without_ext] = boxes[0]
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
                prob_cls_cc = F.softmax(out_cls_cc, dim=1).cpu().numpy().squeeze()
                pred_cls_cc = np.argmax(prob_cls_cc)
                prob_cls_mlo = F.softmax(out_cls_mlo, dim=1).cpu().numpy().squeeze()
                pred_cls_mlo = np.argmax(prob_cls_mlo)
                prob_match = F.softmax(out_match, dim=1).cpu().numpy().squeeze()
                matching_score = prob_match[1]

            match_key = pair_data['key']
            is_mass_cc = (pred_cls_cc < 3) and (prob_cls_cc[pred_cls_cc] >= CONFIDENCE_THRESHOLD)
            is_mass_mlo = (pred_cls_mlo < 3) and (prob_cls_mlo[pred_cls_mlo] >= CONFIDENCE_THRESHOLD)
            is_eligible = is_mass_cc and is_mass_mlo

            if match_key not in results:
                results[match_key] = {'best_score': -1.0, 'final_cc_proposal': None, 
                                      'predicted_class': -1, 'predicted_box_coords': None}

            if is_eligible and matching_score > results[match_key]['best_score']:
                results[match_key]['best_score'] = matching_score
                results[match_key]['final_cc_proposal'] = pair_data['cc']
                cc_pred_name = os.path.splitext(pair_data['cc']['filename'])[0]
                results[match_key]['predicted_box_coords'] = self.yolo_box_map.get(cc_pred_name, None)
                results[match_key]['predicted_class'] = pred_cls_cc
        
        return results

# ---------------------------
# 评估函数 (包含 mAP50 & mAP@[0.5:0.95])
# ---------------------------
def evaluate_results(final_predictions, gt_base_dir, iou_threshold=0.1):
    """
    评估函数：计算 TP/FP/FN/TN/Precision/Recall/F1/Accuracy。
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0 # <-- 新增: True Negatives
    
    # 假设 num_classes=4 (背景类 ID = 3)

    for match_key, prediction_data in final_predictions.items():
        predicted_box_coords = prediction_data.get('predicted_box_coords')
        predicted_class_id = prediction_data['predicted_class']
        is_mass_prediction = predicted_class_id < 3

        cc_prop_info = prediction_data['final_cc_proposal']

        # --- 1. GT 检查 (获取 Ground Truth 状态) ---
        
        # 构造 GT 路径 (使用 match_key 或 cc_prop_info)
        if cc_prop_info is None:
            # 如果没有成功预测 (None)，则从 match_key 获取 GT 文件名
            original_base_name = "_".join(match_key.split('_')) + '_CC'
        else:
            # 如果有成功预测，则从文件名中获取原始名
            original_base_name = "_".join(cc_prop_info['filename'].split('_')[:3])
            
        cc_gt_path = os.path.join(gt_base_dir, 'cc_view', 'labels', 'test', original_base_name + '.txt')
        gt_boxes_data = load_boxes_from_file(cc_gt_path)
        has_ground_truth = len(gt_boxes_data) > 0
        
        # --- 2. 预测与统计 ---
        is_true_positive = False
        
        # A. 如果有成功预测，检查 IoU (TP/FP)
        if cc_prop_info is not None and predicted_box_coords is not None:
            
            if is_mass_prediction and has_ground_truth:
                # 检查 IoU
                pred_box_norm = yolo_to_norm_corners(predicted_box_coords)
                for gt_box_with_class in gt_boxes_data:
                    gt_box_norm = yolo_to_norm_corners(gt_box_with_class)
                    if calculate_iou(pred_box_norm, gt_box_norm) >= iou_threshold:
                        is_true_positive = True
                        break
            
            # 统计 TP 和 FP (基于是否是 Mass 预测)
            if is_mass_prediction and is_true_positive:
                total_tp += 1
            elif is_mass_prediction and not is_true_positive:
                total_fp += 1
        
        # B. 统计 FN 和 TN (评估系统的“错失”和“正确拒绝”)
        
        # Case 1: False Negative (FN) - 模型说没有，但 GT 说有
        # 注意: 只有当 TP 没发生，且 GT 存在时，才计入 FN
        if has_ground_truth and not is_true_positive:
             total_fn += len(gt_boxes_data) # 遗漏的 GT 全部计为 FN
            
        # Case 2: True Negative (TN) - 模型说没有，且 GT 说没有
        # TN 的统计前提是：系统最终没有预测 Mass，且 GT 文件是空的。
        # 如果模型没有选出合格的 proposal (cc_prop_info is None)，或者最终预测是背景，且 GT 是空的
        if not is_mass_prediction and not has_ground_truth:
             total_tn += 1


    # --- 最终指标计算 ---
    
    # 确保分母不为零
    total_detections = total_tp + total_fp
    total_ground_truth = total_tp + total_fn
    total_samples = total_tp + total_fp + total_fn + total_tn

    # Precision (查准率)
    precision = total_tp / total_detections if total_detections > 0 else 0
    # Recall (查全率)
    recall = total_tp / total_ground_truth if total_ground_truth > 0 else 0
    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # Accuracy (准确率)
    accuracy = (total_tp + total_tn) / total_samples if total_samples > 0 else 0

    return {
        'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'TN': total_tn,
        'Total_Samples': total_samples,
        'Precision': precision, 'Recall': recall, 'F1_Score': f1_score,
        'Accuracy': accuracy # <-- 新增 Accuracy
    }


# ---------------------------
# 主函数
# ---------------------------
def main_test_pipeline(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    main_test_pipeline(BEST_MODEL_PATH)
