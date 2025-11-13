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
# from utils.utils import calculate_iou, yolo_to_pixel, load_boxes_from_file, cvtColor, preprocess_input
# (此处省略辅助函数导入，请确保它们可用)

# --- 辅助函数占位符 (用于演示逻辑，请替换为您的实际函数) ---
def load_boxes_from_file(filepath):
    # 此函数载入原始 GT 标签文件 (e.g., [class, x, y, w, h])
    boxes = []
    if not os.path.exists(filepath): return boxes
    with open(filepath, 'r') as f:
        for line in f.readlines():
            parts = [float(p) for p in line.strip().split()]
            if len(parts) >= 5: boxes.append(parts)
    return boxes

def yolo_to_pixel(box, img_width, img_height):
    # 此函数用于获取归一化坐标 [x, y, w, h]
    return box[1:5] # 假设返回 [x_center, y_center, w, h]

def calculate_iou(box1, box2):
    # 此函数用于 IoU 比较
    # (!! 警告: 请使用您之前编写的完整 IoU 实现 !!)
    return 0.0 if np.random.rand() < 0.5 else 1.0 # 占位符，请勿使用
# -------------------------------------------------------------

# --- 配置 (Configuration) ---
BEST_MODEL_PATH = 'logs/best_model.pth' 
TEST_PATCH_DIR = '/kaggle/working/processed_data/test'   # 经过处理的测试集 patches 基础路径 (包含 positive/negative)
GT_LABEL_DIR = '/kaggle/input/breast/data'              # 原始 GT 标签基础路径
YOLO_PRED_LABEL_DIR = '/kaggle/input/siamesedata/siameseD' # 原始 YOLO 预测标签基础路径 (用于获取预测坐标)
INPUT_SHAPE = [64, 64]
NUM_CLASSES = 4 # 3 Positive Classes + 1 Background Class (Class ID = 3)
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5 # CMCNet 预测非背景的最低置信度


class FullDetectionPipeline:
    def __init__(self, model, gt_base_dir, patch_base_dir):
        self.net = model.eval()
        self.gt_base_dir = gt_base_dir
        self.patch_base_dir = patch_base_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.cc_proposals = self._get_all_proposals('cc')
        self.mlo_proposals = self._get_all_proposals('mlo')
        self.all_test_pairs = self._generate_cross_view_pairs()
        print(f"Total {len(self.all_test_pairs)} cross-view proposal pairs generated.")
        
        # 预加载所有 YOLO 预测标签 (用于在 run_inference 中查找坐标)
        self.yolo_box_map = self._load_all_yolo_boxes()

    def _get_all_proposals(self, view):
        """扫描所有 processed proposals (positive/negative)"""
        proposals = {}
        for folder in ['positive', 'negative']:
            path = os.path.join(self.patch_base_dir, view, folder, '*.jpg')
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
        """加载原始 YOLO 预测坐标，用于 IoU 检查 (这是最复杂的查找步骤)"""
        yolo_box_map = {}
        
        # 假设我们将 'test' 文件夹下的所有 YOLO 预测合并起来
        for view in ['cc', 'mlo']:
            pred_dir = os.path.join(YOLO_PRED_LABEL_DIR, 'test', view, 'labels')
            
            # 搜索所有预测文件 (*_0.txt, *_1.txt, etc.)
            for file_path in glob.glob(os.path.join(pred_dir, '*_*.txt')):
                filename = os.path.basename(file_path)
                # Key: '0037..._L_CC_0' (用于与 patch 文件名匹配)
                name_without_ext = os.path.splitext(filename)[0]
                
                # 假设每个 .txt 文件只有一个 box
                boxes = load_boxes_from_file(file_path)
                if boxes:
                    # 存储 [class_id, x, y, w, h]
                    yolo_box_map[name_without_ext] = boxes[0] 
        return yolo_box_map
    
    def _generate_cross_view_pairs(self):
        # (生成所有可能的交叉视图配对)
        all_pairs = []
        for match_key in self.cc_proposals.keys():
            if match_key in self.mlo_proposals:
                for cc_prop in self.cc_proposals[match_key]:
                    for mlo_prop in self.mlo_proposals[match_key]:
                        all_pairs.append({'key': match_key, 'cc': cc_prop, 'mlo': mlo_prop})
        return all_pairs

    def _load_and_preprocess(self, path):
        # (加载和预处理，保持与训练时一致)
        # (略去具体实现)
        return torch.randn(1, 3, self.input_shape[0], self.input_shape[1]).to(self.device) # 占位符

    def run_inference(self):
        """运行 CMCNet 过滤，并找到每个 Patient 的最佳匹配。"""
        results = {}
        
        for pair_data in tqdm(self.all_test_pairs, desc="Running CMCNet Inference"):
            
            img_cc_tensor = self._load_and_preprocess(pair_data['cc']['path'])
            img_mlo_tensor = self._load_and_preprocess(pair_data['mlo']['path'])
            
            with torch.no_grad():
                # 运行模型
                out_cls_cc, out_cls_mlo, out_match, _, _ = self.net(img_cc_tensor, img_mlo_tensor)
                
                # 获取概率和预测
                prob_cls_cc = F.softmax(out_cls_cc, dim=1).cpu().numpy().squeeze()
                pred_cls_cc = np.argmax(prob_cls_cc)
                
                prob_cls_mlo = F.softmax(out_cls_mlo, dim=1).cpu().numpy().squeeze()
                pred_cls_mlo = np.argmax(prob_cls_mlo)
                
                prob_match = F.softmax(out_match, dim=1).cpu().numpy().squeeze()
                matching_score = prob_match[1] # 匹配的概率 (Y=1)

            match_key = pair_data['key']
            
            # --- 决策逻辑 (两阶段筛选) ---
            
            # 1. 分类筛选 (Classification Filter): 检查是否被预测为 Mass (非背景类, Class 3)
            # 预测为 Mass 的置信度必须高于 CONFIDENCE_THRESHOLD
            is_mass_cc = (pred_cls_cc < 3) and (prob_cls_cc[pred_cls_cc] >= CONFIDENCE_THRESHOLD)
            is_mass_mlo = (pred_cls_mlo < 3) and (prob_cls_mlo[pred_cls_mlo] >= CONFIDENCE_THRESHOLD)
            
            # 只有在两个视图都被预测为 Mass 的情况下，我们才考虑匹配分数
            is_eligible = is_mass_cc and is_mass_mlo

            if match_key not in results:
                results[match_key] = {'best_score': -1.0, 'final_cc_proposal': None, 'predicted_class': -1}
            
            # 2. 最终选择 (Matching Selection): 找到最高匹配分数的那对
            if is_eligible and matching_score > results[match_key]['best_score']:
                # 找到新冠军
                results[match_key]['best_score'] = matching_score
                results[match_key]['final_cc_proposal'] = pair_data['cc']
                # **关键: 保存预测坐标 (通过文件名查找)**
                # 提取 CC proposal 的原始 YOLO 标签文件名 (例如: 0037..._L_CC_0)
                cc_pred_name = os.path.splitext(pair_data['cc']['filename'])[0]
                results[match_key]['predicted_box_coords'] = self.yolo_box_map.get(cc_pred_name.replace('_class', ''), None)
                results[match_key]['predicted_class'] = predicted_cc_class
                
        return results

def evaluate_results(final_predictions, gt_base_dir, iou_threshold=0.5):
    """
    评估函数：计算 TP, FP, FN, Precision, Recall。
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # 获取总的 GT 肿块数 (用于计算 Recall 的分母)
    # 此步骤被忽略，因为需要完整的 GT 图像列表。我们将在循环中计算 FN。

    for match_key, prediction_data in final_predictions.items():
        
        predicted_box_coords = prediction_data.get('predicted_box_coords')
        predicted_class_id = prediction_data['predicted_class']
        
        # 1. 预测判断: 预测是否是 Mass (非背景类)
        is_mass_prediction = (predicted_class_id < 3)
        
        # 2. GT 路径: 获取原始 GT 标签文件 (例如: 0037..._L_CC.txt)
        original_base_name = "_".join(cc_prop_info['filename'].split('_')[:3])
        cc_gt_path = os.path.join(gt_base_dir, 'cc_view', 'labels', 'test', original_base_name + '.txt')
        
        # 3. 加载 GT
        gt_boxes_data = load_boxes_from_file(cc_gt_path)
        has_ground_truth = len(gt_boxes_data) > 0 # 检查 GT 文件是否有内容

        # 4. IoU 检查
        is_true_positive = False
        if is_mass_prediction and has_ground_truth and predicted_box_coords is not None:
            # IoU 验证: predicted_box_coords 是 [class_id, x, y, w, h]
            pred_box = predicted_box_coords[1:] # [x, y, w, h]
            
            for gt_box_with_class in gt_boxes_data:
                gt_box = gt_box_with_class[1:] # [x, y, w, h]
                
                iou_score = calculate_iou(pred_box, gt_box)
                if iou_score >= iou_threshold:
                    is_true_positive = True
                    break

        # 5. 统计 TP, FP, FN
        if is_mass_prediction and is_true_positive:
            total_tp += 1
        elif is_mass_prediction and not is_true_positive:
            total_fp += 1
        elif not is_mass_prediction and has_ground_truth:
            total_fn += 1 # 遗漏了一个 GT 肿块

    # (此处的总 FN 统计是不完整的，但用于估算)
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
