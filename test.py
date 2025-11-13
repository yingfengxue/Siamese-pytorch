import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import math

# --- 导入您的模型和工具 (请根据您系统的实际位置修改) ---
from nets.siamese import CMCNet
from utils.utils import cvtColor
# (确保 utils.py 中的 cvtColor, preprocess_input 等函数可用)

# --- 辅助函数 ---

def load_boxes_from_file(filepath):
    """载入 YOLO 标签文件 (e.g., [class, x, y, w, h])"""
    boxes = []
    if not os.path.exists(filepath): return boxes
    with open(filepath, 'r') as f:
        for line in f.readlines():
            parts = [float(p) for p in line.strip().split()]
            if len(parts) >= 5: boxes.append(parts)
    return boxes

def yolo_to_norm_corners(box):
    """将 YOLO 格式 [class, x_center, y_center, w, h] 转换为归一化角点坐标 [x1, y1, x2, y2]"""
    # 假设 box 是 [class, x_center, y_center, w, h]
    x_center, y_center, w, h = box[1:5]
    x1 = max(0, x_center - w / 2)
    y1 = max(0, y_center - h / 2)
    x2 = min(1, x_center + w / 2)
    y2 = min(1, y_center + h / 2)
    return [x1, y1, x2, y2]

def calculate_iou(boxA, boxB):
    """计算两个边界框 (格式: [x1, y1, x2, y2]) 的 IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    unionArea = boxAArea + boxBArea - interArea + 1e-6
    return interArea / unionArea

# --- 配置 (Configuration) ---
BEST_MODEL_PATH = '/kaggle/working/logs/ep020-loss1.689-val1.724.pth'
TEST_PATCH_DIR = '/kaggle/input/siamesedata/siameseD/test'
GT_LABEL_DIR = '/kaggle/input/breast/data'
YOLO_PRED_LABEL_DIR = '/kaggle/input/siamesedata/siameseD'
INPUT_SHAPE = [224, 224]
NUM_CLASSES = 4 # 3 Positive Classes + 1 Background Class (Class ID = 3)
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5 # CMCNet 预测非背景的最低置信度


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
        """扫描原始 YOLO 裁剪输出 (RAW YOLO CROPS)"""
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
        """加载原始 YOLO 预测坐标，用于 IoU 检查"""
        yolo_box_map = {}
        for view in ['cc', 'mlo']:
            # 查找 'test/cc/labels' 目录下的所有预测文件
            pred_dir = os.path.join(self.yolo_pred_dir, 'test', view, 'labels')
            for file_path in glob.glob(os.path.join(pred_dir, '*_*.txt')):
                filename = os.path.basename(file_path)
                name_without_ext = os.path.splitext(filename)[0]
                boxes = load_boxes_from_file(file_path)
                if boxes:
                    # 存储 [class_id, x, y, w, h]
                    yolo_box_map[name_without_ext] = boxes[0]
        return yolo_box_map

    def _generate_cross_view_pairs(self):
        """生成所有可能的交叉视图配对 (Cartesian Product)"""
        all_pairs = []
        for match_key in self.cc_proposals.keys():
            if match_key in self.mlo_proposals:
                for cc_prop in self.cc_proposals[match_key]:
                    for mlo_prop in self.mlo_proposals[match_key]:
                        all_pairs.append({'key': match_key, 'cc': cc_prop, 'mlo': mlo_prop})
        return all_pairs

    def _load_and_preprocess(self, path):
        """加载图像并进行预处理"""
        image = Image.open(path).convert('RGB')
        # 尺寸必须与训练时一致
        image = image.resize((self.input_shape[1], self.input_shape[0])) 
        # 简化预处理 (请确保与您的训练预处理保持一致)
        image = np.array(image, dtype=np.float32) / 255.0
        # 调整轴序 (H, W, C) -> (C, H, W)
        image = image.transpose(2, 0, 1) 
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        return tensor

    def run_inference(self):
        """运行 CMCNet 过滤，并找到每个 Patient 的最佳匹配。"""
        results = {}
        for pair_data in tqdm(self.all_test_pairs, desc="CMCNet Inference"):
            img_cc = self._load_and_preprocess(pair_data['cc']['path'])
            img_mlo = self._load_and_preprocess(pair_data['mlo']['path'])
            
            with torch.no_grad():
                out_cls_cc, out_cls_mlo, out_match, _, _ = self.net(img_cc, img_mlo)
                
                # 获取概率和预测 (CPU/Numpy)
                prob_cls_cc = F.softmax(out_cls_cc, dim=1).cpu().numpy().squeeze()
                pred_cls_cc = np.argmax(prob_cls_cc)
                prob_cls_mlo = F.softmax(out_cls_mlo, dim=1).cpu().numpy().squeeze()
                pred_cls_mlo = np.argmax(prob_cls_mlo)
                prob_match = F.softmax(out_match, dim=1).cpu().numpy().squeeze()
                matching_score = prob_match[1] # 匹配的概率 (Y=1)

            match_key = pair_data['key']
            
            # --- 决策逻辑 (两阶段筛选) ---
            # 1. 分类筛选: 检查是否被预测为 Mass (非背景类, Class 3)
            is_mass_cc = (pred_cls_cc < 3) and (prob_cls_cc[pred_cls_cc] >= CONFIDENCE_THRESHOLD)
            is_mass_mlo = (pred_cls_mlo < 3) and (prob_cls_mlo[pred_cls_mlo] >= CONFIDENCE_THRESHOLD)
            is_eligible = is_mass_cc and is_mass_mlo

            if match_key not in results:
                # 初始化结果字典，添加一个 None 来确保后续可以正确统计 FN
                results[match_key] = {'best_score': -1.0, 'final_cc_proposal': None, 'predicted_class': -1, 'predicted_box_coords': None}

            # 2. 最终选择: 找到最高匹配分数的那对 (在 eligible 的 Proposals 中)
            if is_eligible and matching_score > results[match_key]['best_score']:
                results[match_key]['best_score'] = matching_score
                results[match_key]['final_cc_proposal'] = pair_data['cc']
                
                # 关键: 关联坐标
                cc_pred_name = os.path.splitext(pair_data['cc']['filename'])[0]
                # 原始 YOLO 预测文件名格式: ID_SIDE_VIEW_INDEX
                # 我们需要找到匹配的 YOLO box
                results[match_key]['predicted_box_coords'] = self.yolo_box_map.get(cc_pred_name, None)
                results[match_key]['predicted_class'] = pred_cls_cc
        
        return results

def evaluate_results(final_predictions, gt_base_dir, iou_threshold=0.5):
    """
    评估函数：计算 TP, FP, FN, Precision, Recall, F1_Score (代替 Accuracy)。
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for match_key, prediction_data in final_predictions.items():
        predicted_box_coords = prediction_data.get('predicted_box_coords')
        predicted_class_id = prediction_data['predicted_class']
        cc_prop_info = prediction_data['final_cc_proposal']

        # --- 1. GT 检查 ---
        
        # 如果该患者没有任何被选中的 proposal
        if cc_prop_info is None:
            # 尝试加载所有 GT，统计为 FN
            # 构造 GT 路径: match_key = ID_SIDE -> ID_SIDE_CC.txt
            original_base_name = "_".join(match_key.split('_')) + '_CC' 
            cc_gt_path = os.path.join(gt_base_dir, 'cc_view', 'labels', 'test', original_base_name + '.txt')
            gt_boxes_data = load_boxes_from_file(cc_gt_path)
            total_fn += len(gt_boxes_data)  # 所有 GT 都算作 FN
            continue

        # --- 2. 预测有效性 ---
        is_mass_prediction = predicted_class_id < 3

        # 3. 加载 GT
        original_base_name = "_".join(cc_prop_info['filename'].split('_')[:3])
        cc_gt_path = os.path.join(gt_base_dir, 'cc_view', 'labels', 'test', original_base_name + '.txt')
        gt_boxes_data = load_boxes_from_file(cc_gt_path)
        has_ground_truth = len(gt_boxes_data) > 0

        # 4. IoU 检查
        is_true_positive = False
        if is_mass_prediction and has_ground_truth and predicted_box_coords is not None:
            pred_box_norm = yolo_to_norm_corners(predicted_box_coords)
            
            for gt_box_with_class in gt_boxes_data:
                gt_box_norm = yolo_to_norm_corners(gt_box_with_class)
                if calculate_iou(pred_box_norm, gt_box_norm) >= iou_threshold:
                    is_true_positive = True
                    break

        # 5. 统计 TP, FP, FN
        if is_mass_prediction and is_true_positive:
            total_tp += 1
        elif is_mass_prediction and not is_true_positive:
            total_fp += 1
        elif not is_mass_prediction and has_ground_truth:
            total_fn += len(gt_boxes_data) # 遗漏的 GT 全部计为 FN

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'Precision': precision, 'Recall': recall, 'F1_Score': f1_score}


# --- 最终运行函数 ---
def main_test_pipeline(model_path):
    from nets.siamese import CMCNet
    
    # 1. 初始化模型
    model = CMCNet(num_classes=NUM_CLASSES, pretrained=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 2. 初始化 Pipeline
    pipeline = FullDetectionPipeline(model, GT_LABEL_DIR, TEST_PATCH_DIR, YOLO_PRED_LABEL_DIR)
    
    # 3. 运行推理和筛选
    final_predictions = pipeline.run_inference()
    
    # 4. 评估
    evaluation_results = evaluate_results(final_predictions, GT_LABEL_DIR) 
    
    print("\n--- 最终评估结果 (Testing Metrics) ---")
    for k, v in evaluation_results.items():
        print(f"{k}: {v}")
        
if __name__ == '__main__':
    main_test_pipeline(BEST_MODEL_PATH)
