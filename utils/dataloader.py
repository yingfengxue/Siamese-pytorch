import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import cv2
import os
import glob
import random

# --- (从您之前的代码中复制的辅助函数) ---
# --- (我们重用所有的数据增强和预处理逻辑) ---

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image

class Resize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize(self.size, self.interpolation)

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # (假设您有这个类的实现)
        # 简单的实现：
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=Image.BICUBIC):
        # (这是一个简化的实现，您应该使用您自己的)
        self.size = size
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        # (简化逻辑)
        return img.resize(self.size, self.interpolation)

class ImageNetPolicy(object):
    def __init__(self):
        # (假设您有这个类的实现)
        pass
    def __call__(self, img):
        # (返回原始图像作为占位符)
        return img

# --- (这是您的新 Dataloader) ---

class MultiTaskDataset(Dataset):
    def __init__(self, data_dir, input_shape, num_classes, random, autoaugment_flag=True):
        super(MultiTaskDataset, self).__init__()
        
        self.input_shape = input_shape
        self.random = random
        
        # 1. 定义您的类别
        self.num_classes = num_classes
        # "背景" (Negative) 类别将被分配为 num_classes
        self.background_class_id = num_classes 

        # 2. 扫描文件并建立“匹配对”
        print(f"Loading data from: {data_dir}")
        self.positive_pairs = []
        self.all_positive_cc_map = {} # 用于创建 "不匹配" 对
        self.all_positive_mlo_map = {} # 用于创建 "不匹配" 对
        
        cc_pos_files = glob.glob(os.path.join(data_dir, 'cc', 'positive', '*.jpg'))
        mlo_pos_files = glob.glob(os.path.join(data_dir, 'mlo', 'positive', '*.jpg'))
        
        self.cc_neg_files = glob.glob(os.path.join(data_dir, 'cc', 'negative', '*.jpg'))
        self.mlo_neg_files = glob.glob(os.path.join(data_dir, 'mlo', 'negative', '*.jpg'))

        # --- 匹配逻辑 ---
        # A. 将所有CC阳性样本读入一个Map中，使用“匹配键”
        cc_pos_map = {}
        for f_cc in cc_pos_files:
            try:
                key, class_id = self._parse_positive_filename(os.path.basename(f_cc))
                cc_pos_map[key] = {"path": f_cc, "class": class_id}
                self.all_positive_cc_map[f_cc] = class_id
            except Exception as e:
                print(f"Warning: Skipping CC file (could not parse): {f_cc}. Error: {e}")

        # B. 遍历MLO阳性样本，并在CC Map中查找匹配项
        for f_mlo in mlo_pos_files:
            try:
                key, class_id = self._parse_positive_filename(os.path.basename(f_mlo))
                self.all_positive_mlo_map[f_mlo] = class_id
                
                # 如果找到了匹配的CC键
                if key in cc_pos_map:
                    cc_match = cc_pos_map[key]
                    self.positive_pairs.append({
                        "cc_path": cc_match["path"],
                        "cc_class": cc_match["class"],
                        "mlo_path": f_mlo,
                        "mlo_class": class_id
                    })
            except Exception as e:
                print(f"Warning: Skipping MLO file (could not parse): {f_mlo}. Error: {e}")
        
        # 用于采样的完整列表
        self.all_positive_cc_list = list(self.all_positive_cc_map.keys())
        self.all_positive_mlo_list = list(self.all_positive_mlo_map.keys())

        print(f"Found {len(self.positive_pairs)} matching positive pairs.")
        print(f"Found {len(self.cc_neg_files)} CC negative patches.")
        print(f"Found {len(self.mlo_neg_files)} MLO negative patches.")

        # 3. 设置数据增强 (重用您代码中的逻辑)
        # 3. 设置数据增强
        self.autoaugment_flag = autoaugment_flag
        
        # --- (!! 关键修复 !!) ---
        # 无论 autoaugment_flag 是什么，我们都需要为 random=True (训练) 和 random=False (验证) 
        # 两种情况初始化 *所有* 需要的变换。
        
        # (1) 用于 autoaugment (random=True)
        self.resize_crop = RandomResizedCrop(input_shape)
        self.policy = ImageNetPolicy()
        
        # (2) 用于 validation (random=False)
        # (这在 autoaugment_flag=True 时被错误地跳过了)
        self.resize = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
        self.center_crop = CenterCrop(input_shape)

    def _parse_positive_filename(self, filename):
        """
        解析文件名，如 "0037..._L_CC_gt_0_class0.jpg"
        返回一个唯一的匹配键 "0037..._L_gt_0" 和 类别 0
        """
        name_no_ext = os.path.splitext(filename)[0]
        
        # 分离类别
        parts = name_no_ext.split('_class')
        class_id = int(parts[-1])
        
        # 构建匹配键
        key_part = parts[0] # e.g., "0037..._L_CC_gt_0"
        key_parts = key_part.split('_')
        
        patient_id = key_parts[0] # "0037..."
        side = key_parts[1]       # "L"
        gt_index = "_".join(key_parts[3:]) # "gt_0"
        
        # 键 "0037..._L_gt_0"
        key = f"{patient_id}_{side}_{gt_index}"
        
        if class_id >= self.num_classes:
             raise ValueError(f"File {filename} has class_id {class_id} which is >= num_classes {self.num_classes}")

        return key, class_id

    def __len__(self):
        # 让我们定义一个epoch的大小为：所有匹配对 + 所有阴性patches
        return len(self.positive_pairs) + len(self.cc_neg_files) + len(self.mlo_neg_files)

    def __getitem__(self, index):
        
        # --- 1. 采样策略：决定是生成“匹配对”还是“不匹配对” ---
        # 50% 几率生成一个“真匹配对”
        if index < len(self.positive_pairs) and random.random() < 0.5:
            # --- (A) 生成一个“匹配” (Match = 1) 样本 ---
            
            pair_info = self.positive_pairs[index % len(self.positive_pairs)]
            
            patch_cc_path = pair_info["cc_path"]
            patch_mlo_path = pair_info["mlo_path"]
            label_cls_cc = pair_info["cc_class"]
            label_cls_mlo = pair_info["mlo_class"]
            label_match = 1.0 # 匹配
            
        else:
            # --- (B) 生成一个“不匹配” (Match = 0) 样本 ---
            label_match = 0.0 # 不匹配
            
            # 随机决定不匹配的类型
            rand_type = random.random()
            
            if rand_type < 0.33:
                # 1. (Pos CC, Neg MLO)
                patch_cc_path = random.choice(self.all_positive_cc_list)
                patch_mlo_path = random.choice(self.mlo_neg_files)
                
                label_cls_cc = self.all_positive_cc_map[patch_cc_path]
                label_cls_mlo = self.background_class_id
                
            elif rand_type < 0.66:
                # 2. (Neg CC, Pos MLO)
                patch_cc_path = random.choice(self.cc_neg_files)
                patch_mlo_path = random.choice(self.all_positive_mlo_list)

                label_cls_cc = self.background_class_id
                label_cls_mlo = self.all_positive_mlo_map[patch_mlo_path]
                
            else:
                # 3. (Neg CC, Neg MLO)
                patch_cc_path = random.choice(self.cc_neg_files)
                patch_mlo_path = random.choice(self.mlo_neg_files)
                
                label_cls_cc = self.background_class_id
                label_cls_mlo = self.background_class_id

        # --- 2. 加载和处理图像 ---
        
        # 加载 CC Patch
        image_cc = Image.open(patch_cc_path)
        image_cc = cvtColor(image_cc)
        if self.autoaugment_flag:
            image_cc = self.AutoAugment(image_cc, random=self.random)
        else:
            image_cc = self.get_random_data(image_cc, self.input_shape, random=self.random)
        image_cc = preprocess_input(np.array(image_cc).astype(np.float32))
        image_cc = np.transpose(image_cc, [2, 0, 1])
        
        # 加载 MLO Patch
        image_mlo = Image.open(patch_mlo_path)
        image_mlo = cvtColor(image_mlo)
        if self.autoaugment_flag:
            image_mlo = self.AutoAugment(image_mlo, random=self.random)
        else:
            image_mlo = self.get_random_data(image_mlo, self.input_shape, random=self.random)
        image_mlo = preprocess_input(np.array(image_mlo).astype(np.float32))
        image_mlo = np.transpose(image_mlo, [2, 0, 1])

        # --- 3. 返回所有数据 ---
        return (
            torch.from_numpy(image_cc).type(torch.FloatTensor),
            torch.from_numpy(image_mlo).type(torch.FloatTensor),
            torch.tensor(label_cls_cc, dtype=torch.long),
            torch.tensor(label_cls_mlo, dtype=torch.long),
            torch.tensor(label_match, dtype=torch.float32)
        )

    # --- (重用您所有的增强函数) ---

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            return image_data

        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15, 15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128, 128, 128]) 

        image_data = np.array(image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data
    
    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        image = self.resize_crop(image)
        
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        image = self.policy(image)
        return image

# --- (您的新 Collate Function) ---

def multitask_collate(batch):
    """
    这个 collate_fn 会将 Dataloader 的输出整理成 (inputs), (labels) 的格式
    """
    
    # batch 是一个列表，
    # 每一项是 (image_cc, image_mlo, label_cls_cc, label_cls_mlo, label_match)
    
    images_cc = torch.stack([item[0] for item in batch], dim=0)
    images_mlo = torch.stack([item[1] for item in batch], dim=0)
    
    labels_cls_cc = torch.stack([item[2] for item in batch], dim=0)
    labels_cls_mlo = torch.stack([item[3] for item in batch], dim=0)
    
    # (B, 1)
    labels_match = torch.stack([item[4] for item in batch], dim=0).unsqueeze(1) 
    
    # 组合成 (inputs), (labels)
    # inputs 是一个 (cc_batch, mlo_batch) 的元组
    # labels 是一个 (cc_labels, mlo_labels, match_labels) 的元组
    return (images_cc, images_mlo), (labels_cls_cc, labels_cls_mlo, labels_match)
