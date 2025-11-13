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
TEST_PATCH_DIR = '/kaggle/input/siamesedata/siameseD/test'
GT_LABEL_DIR = '/kaggle/input/breast/data'
YOLO_PRED_LABEL_DIR = '/kaggle/input/siamesedata/siameseD'
INPUT_SHAPE = [224, 224]
NUM_CLASSES = 4
CONFIDENCE_THRESHOLD = 0.5

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
from mean_average_precision import MetricBuilder
import numpy as np
import os

def evaluate_results(final_predictions, gt_base_dir, num_classes=4):
    preds_list = []
    gt_list = []
    total_tp = total_fp = total_fn = 0

    # 给每个 match_key 分配唯一整数 ID (map_2d 要求 image_id)
    match_key2id = {k: idx for idx, k in enumerate(final_predictions.keys())}

    for match_key, data in final_predictions.items():
        pred_class = data['predicted_class']
        pred_box = data['predicted_box_coords']
        cc_prop = data['final_cc_proposal']

        image_id = match_key2id[match_key]

        # ------------------
        # Ground Truth
        # ------------------
        gt_path = os.path.join(gt_base_dir, 'cc_view', 'labels', 'test', match_key + '_CC.txt')
        gt_boxes = load_boxes_from_file(gt_path)
        for gt in gt_boxes:
            x1, y1, x2, y2 = yolo_to_norm_corners(gt)
            difficult = 0  # 默认非难例
            gt_list.append([image_id, gt[0], x1, y1, x2, y2, difficult])

        # ------------------
        # 预测
        # ------------------
        if cc_prop is not None and pred_class < 3 and pred_box is not None:
            x1, y1, x2, y2 = yolo_to_norm_corners(pred_box)
            score = data.get('best_score', 1.0)
            preds_list.append([image_id, pred_class, x1, y1, x2, y2, score])

            # TP/FP/FN 统计
            is_tp = any(calculate_iou([x1, y1, x2, y2], yolo_to_norm_corners(gt)) >= 0.5 for gt in gt_boxes)
            if is_tp:
                total_tp += 1
            else:
                total_fp += 1
        else:
            total_fn += len(gt_boxes)

    # ------------------
    # Precision / Recall / F1
    # ------------------
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1_score = 2*precision*recall/(precision+recall) if (precision+recall) else 0

    # ------------------
    # mAP 计算
    # ------------------
    preds = np.array(preds_list, dtype=np.float32)
    gts = np.array(gt_list, dtype=np.float32)

    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=num_classes)
    metric_fn.add(preds, gts)
    res_all = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05))
    res50 = metric_fn.value(iou_thresholds=[0.5])

    return {
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1_score,
        'mAP50': res50['mAP'],
        'mAP@[0.5:0.95]': res_all['mAP']
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
