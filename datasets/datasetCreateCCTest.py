import cv2
import numpy as np
import os
import glob

# --- 1. 配置您的路径和参数 ---
VIEW = 'cc'      
SPLIT = 'test'

# --- 调整K值 (每张图像保留的阴性样本数) ---
K_NEGATIVE_PATCHES = 5 
IOU_THRESHOLD = 0.5


# --- 2. 您的具体路径 (直接指定) ---
# (这是为 'cc' 视图 和 'test' 数据集设置的)

# 2a. 原始全尺寸图像
FULL_IMAGE_DIR = '/kaggle/input/breast/data/cc_view/images/test'

# 2b. 原始 Ground Truth 标签
GT_LABEL_DIR = '/kaggle/input/breast/data/cc_view/labels/test'

# 2c. 您的YOLO *预测* 标签
YOLO_PRED_LABEL_DIR = '/kaggle/input/siamesedata/siameseD/test/cc/labels' 

# 2d. 输出路径 (!! 重要 !!)
# Kaggle的 /kaggle/input 是只读的, 您必须写入 /kaggle/working/

OUTPUT_DIR = '/kaggle/working/processed_data/test/cc'
POSITIVE_PATCH_DIR = os.path.join(OUTPUT_DIR, 'positive')
NEGATIVE_PATCH_DIR = os.path.join(OUTPUT_DIR, 'negative')


# --- 3. 辅助函数 ---

def yolo_to_pixel(box, img_width, img_height):
    """将YOLO格式 [x_center, y_center, w, h] (normalized) 转换为 [x1, y1, x2, y2] (pixel)"""
    x_center, y_center, w, h = box
    x1 = int((x_center - w / 2) * img_width)
    y1 = int((y_center - h / 2) * img_height)
    x2 = int((x_center + w / 2) * img_width)
    y2 = int((y_center + h / 2) * img_height)
    # 确保坐标在图像范围内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width - 1, x2)
    y2 = min(img_height - 1, y2)
    return x1, y1, x2, y2

def calculate_iou(boxA, boxB):
    """计算两个边界框 [x1, y1, x2, y2] 的 IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # +1 是因为像素坐标是包含的
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def load_boxes_from_file(filepath):
    """从YOLO txt文件中加载边界框。假设格式: class x y w h [confidence]"""
    boxes = []
    if not os.path.exists(filepath):
        return boxes
    with open(filepath, 'r') as f:
        for line in f.readlines():
            parts = [float(p) for p in line.strip().split()]
            if len(parts) < 5: continue 
            box_data = {
                "class_id": int(parts[0]),
                "box_yolo": parts[1:5],
                "confidence": parts[5] if len(parts) > 5 else 1.0 
            }
            boxes.append(box_data)
    return boxes

# --- 4. 主处理脚本 ---

def main():
    print("--- 开始处理 ---")
    print(f"视图 (VIEW): {VIEW}")
    print(f"数据 (SPLIT): {SPLIT}")
    
    # 检查输入路径
    if not os.path.isdir(FULL_IMAGE_DIR):
        print(f"*** 错误: 图像路径不存在: {FULL_IMAGE_DIR}")
        return
    if not os.path.isdir(GT_LABEL_DIR):
        print(f"*** 错误: GT标签路径不存在: {GT_LABEL_DIR}")
        return
    if not os.path.isdir(YOLO_PRED_LABEL_DIR):
        print(f"*** 错误: YOLO预测标签路径不存在: {YOLO_PRED_LABEL_DIR}")
        return

    # 确保输出目录存在
    os.makedirs(POSITIVE_PATCH_DIR, exist_ok=True)
    os.makedirs(NEGATIVE_PATCH_DIR, exist_ok=True)
    print(f"创建阳性输出目录: {POSITIVE_PATCH_DIR}")
    print(f"创建阴性输出目录: {NEGATIVE_PATCH_DIR}")

    print(f"\n正在读取全尺寸图像: {FULL_IMAGE_DIR}")
    
    image_files = []
    for ext in ('*.jpg', '*.png', '*.jpeg'):
        image_files.extend(glob.glob(os.path.join(FULL_IMAGE_DIR, ext)))

    if not image_files:
        print(f"*** 警告: 在 {FULL_IMAGE_DIR} 中未找到任何图像文件。")
        return

    print(f"共找到 {len(image_files)} 张图像。")
    
    total_pos_patches = 0
    total_neg_patches = 0

    for img_path in image_files:
        img_filename = os.path.basename(img_path)
        base_filename = os.path.splitext(img_filename)[0]
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像 {img_path}, 跳过。")
            continue
        
        img_height, img_width = img.shape[:2]

        # --- 任务 A: 加载阳性 Patches (来自 Ground Truth) ---
        gt_label_path = os.path.join(GT_LABEL_DIR, base_filename + '.txt')
        gt_boxes_data = load_boxes_from_file(gt_label_path)

        gt_pixel_boxes = []
        for gt_data in gt_boxes_data:
            gt_pixel_boxes.append(yolo_to_pixel(gt_data["box_yolo"], img_width, img_height))

        # --- (!! 关键修复 !!) ---
        # --- 任务 B: 加载阴性 Patches (来自 YOLO 假阳性) ---
        
        # 查找所有匹配的YOLO预测文件 (e.g., ..._0.txt, ..._1.txt)
        yolo_label_pattern = os.path.join(YOLO_PRED_LABEL_DIR, base_filename + "_*.txt")
        yolo_label_files = glob.glob(yolo_label_pattern)
        
        yolo_boxes_data = [] # 初始化一个空列表
        
        if not yolo_label_files:
            # 这是一个新加的警告，如果找不到任何YOLO预测文件
            pass # 我们可以跳过警告，因为这可能是正常的
            # print(f"警告: [阴性] 未找到 {base_filename} 对应的YOLO预测文件 (模式: {yolo_label_pattern})")

        # 循环读取所有找到的YOLO预测文件
        for yolo_file in yolo_label_files:
            yolo_boxes_data.extend(load_boxes_from_file(yolo_file))
        
        # --- (修复结束) ---

        # --- 任务 A: 提取阳性 Patches (来自 Ground Truth) ---
        for i, gt_data in enumerate(gt_boxes_data):
            x1, y1, x2, y2 = yolo_to_pixel(gt_data["box_yolo"], img_width, img_height)
            patch = img[y1:y2, x1:x2]
            
            if patch.size == 0:
                print(f"警告: [阳性] 在 {img_filename} 中为 GT {i} 裁切到空 patch，跳过。")
                continue

            class_id = gt_data["class_id"]
            output_filename = f"{base_filename}_gt_{i}_class{class_id}.jpg"
            save_path = os.path.join(POSITIVE_PATCH_DIR, output_filename)
            cv2.imwrite(save_path, patch)
            total_pos_patches += 1

        # --- 任务 B: 提取阴性 Patches (来自 YOLO 假阳性) ---
        false_positives = [] 
        
        # (现在 yolo_boxes_data 包含了所有 ..._0.txt, ..._1.txt 等文件中的内容)
        for yolo_data in yolo_boxes_data:
            yolo_pixel_box = yolo_to_pixel(yolo_data["box_yolo"], img_width, img_height)
            
            max_iou = 0.0
            if not gt_pixel_boxes:
                 max_iou = 0.0
            else:
                for gt_pixel_box in gt_pixel_boxes:
                    iou = calculate_iou(yolo_pixel_box, gt_pixel_box)
                    if iou > max_iou:
                        max_iou = iou

            if max_iou < IOU_THRESHOLD:
                false_positives.append((yolo_data["confidence"], yolo_pixel_box))

        false_positives.sort(key=lambda x: x[0], reverse=True)
        
        for i, (conf, (x1, y1, x2, y2)) in enumerate(false_positives[:K_NEGATIVE_PATCHES]):
            patch = img[y1:y2, x1:x2]

            if patch.size == 0:
                print(f"警告: [阴性] 在 {img_filename} 中为 阴性 {i} 裁切到空 patch，跳过。")
                continue
                
            output_filename = f"{base_filename}_neg_{i}_conf{conf:.2f}.jpg"
            save_path = os.path.join(NEGATIVE_PATCH_DIR, output_filename)
            cv2.imwrite(save_path, patch)
            total_neg_patches += 1

    print("\n--- 处理完成 ---")
    print(f"总共为 {VIEW} - {SPLIT} 保存了 {total_pos_patches} 个阳性 patches。")
    print(f"总共为 {VIEW} - {SPLIT} 保存了 {total_neg_patches} 个阴性 patches。")
    print(f"阳性 patches保存在: {POSITIVE_PATCH_DIR}")
    print(f"阴性 patches保存在: {NEGATIVE_PATCH_DIR}")

# --- 6. 运行脚本 ---
if __name__ == "__main__":
    main()
