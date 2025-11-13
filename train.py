import os
import datetime # <-- 新增
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm # <-- 新增

# --- (!! 关键修改 !!) ---
# 1. 导入您的新模型和 Dataloader
from nets.siamese import CMCNet
from utils.dataloader import MultiTaskDataset, multitask_collate
# -------------------------

from utils.callbacks import LossHistory
from utils.utils import (download_weights, get_lr_scheduler,
                         set_optimizer_lr, show_config)
# from utils.utils import load_dataset # <-- 移除, 不再需要
# from utils.utils_fit import fit_one_epoch # <-- 移除, 我们将重写它

if __name__ == "__main__":
    #----------------------------------------------------#
    #   是否使用Cuda
    #----------------------------------------------------#
    Cuda            = True
    distributed     = False
    sync_bn         = False
    fp16            = False
    
    # --- (!! 关键修改 !!) ---
    # 1. 您的3个 Positive 类别 + 1个 "背景" 类别
    num_positive_classes = 3
    num_classes = 4 
    
    # 2. 您的新数据路径
    train_data_path = "/kaggle/working/processed_data/train"
    val_data_path   = "/kaggle/working/processed_data/val"
    
    # [cite_start]3. 论文中使用的输入大小 [cite: 199]
    input_shape     = [64, 64]
    
    # [cite_start]4. 论文中用于多任务损失的系数 [cite: 175]
    alpha = 1.0 # (for L_cls_cc)
    beta  = 1.0 # (for L_cls_mlo)
    gamma = 1.0 # (for L_mat)
    # -------------------------
    
    pretrained      = True
    model_path      = ""

    #------------------------------------------------------#
    #   训练参数
    #------------------------------------------------------#
    Init_Epoch      = 0
    Epoch           = 100
    batch_size      = 32
    
    #------------------------------------------------------------------#
    #   学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "sgd"
    momentum            = 0.9
    weight_decay        = 5e-4
    lr_decay_type       = 'cos'
    save_period         = 10
    save_dir            = 'logs'
    num_workers         = 4

    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights("vgg16")  
            dist.barrier()
        else:
            download_weights("vgg16")  

    # --- (!! 关键修改 !!) ---
    # 初始化您的新 CMCNet 模型
    model = CMCNet(num_classes=num_classes, pretrained=pretrained)
    # -------------------------

    if model_path != '':
        # (加载权重的逻辑保持不变)
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    
    # --- (!! 关键修改 !!) ---
    # 获得(三个)损失函数
    # 1. & 2. 分类损失 (用于 4-class 输出)
    loss_cls = nn.CrossEntropyLoss()
    # 3. 匹配损失 (用于 2-class 输出: "匹配" vs "不匹配")
    loss_match = nn.CrossEntropyLoss()
    # -------------------------

    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None
        
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # --- (!! 关键修改 !!) ---
    # 移除旧的 load_dataset
    # train_ratio = 0.9
    # train_lines, train_labels, val_lines, val_labels = load_dataset(dataset_path, train_own_data, train_ratio)
    
    # 使用新的 MultiTaskDataset
    train_dataset = MultiTaskDataset(train_data_path, input_shape, num_classes=num_positive_classes, random=True)
    val_dataset   = MultiTaskDataset(val_data_path, input_shape, num_classes=num_positive_classes, random=False)
    
    num_train = len(train_dataset)
    num_val   = len(val_dataset)
    # -------------------------

    if local_rank == 0:
        show_config(
            model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        wanted_step = 3e4 if optimizer_type == "sgd" else 1e4
        total_step  = num_train // batch_size * Epoch
        if total_step <= wanted_step:
            wanted_epoch = wanted_step // (num_train // batch_size) + 1
            print("\n\033[1;33;44m[Warning] ... (您的警告信息) ... \033[0m") # (保留)

    if True:
        # (自适应学习率和优化器设置保持不变)
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        # --- (!! 关键修改 !!) ---
        # 使用新的 Dataloader 和 collate_fn
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=multitask_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=multitask_collate, sampler=val_sampler)
        # -------------------------

        # --- (!! 关键修改 !!) ---
        # 移除 fit_one_epoch，重写训练循环
        
        for epoch in range(Init_Epoch, Epoch):
            if distributed:
                train_sampler.set_epoch(epoch)
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            # --- 开始训练 ---
            total_loss = 0
            total_cls_loss = 0
            total_match_loss = 0
            
            val_loss = 0
            val_cls_loss = 0
            val_match_loss = 0
            
            if local_rank == 0:
                print('Start Train')
                pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
                
            model_train.train()
            for i, batch in enumerate(gen):
                if i >= epoch_step:
                    break
                
                # 1. 获取数据 (cc_batch, mlo_batch), (cc_labels, mlo_labels, match_labels)
                inputs, labels = batch
                images_cc, images_mlo = inputs
                labels_cls_cc, labels_cls_mlo, labels_match = labels
                
                if Cuda:
                    images_cc       = images_cc.cuda(local_rank)
                    images_mlo      = images_mlo.cuda(local_rank)
                    labels_cls_cc   = labels_cls_cc.cuda(local_rank)
                    labels_cls_mlo  = labels_cls_mlo.cuda(local_rank)
                    labels_match    = labels_match.cuda(local_rank)

                # 2. 梯度清零
                optimizer.zero_grad()
                
                # 3. 前向传播
                if not fp16:
                    # (获取3个输出)
                    out_cls_cc, out_cls_mlo, out_match, _, _ = model_train(images_cc, images_mlo)
                    
                    # (计算3个损失)
                    l_cls_cc = loss_cls(out_cls_cc, labels_cls_cc)
                    l_cls_mlo = loss_cls(out_cls_mlo, labels_cls_mlo)
                    # (squeeze labels_match from [B, 1] to [B])
                    l_match = loss_match(out_match, labels_match.squeeze(1).long()) 

                    # (多任务损失) [cite_start][cite: 171]
                    l_total = (alpha * l_cls_cc) + (beta * l_cls_mlo) + (gamma * l_match)
                    
                    # (反向传播)
                    l_total.backward()
                    optimizer.step()
                else:
                    # (FP16 逻辑)
                    from torch.cuda.amp import autocast
                    with autocast():
                        out_cls_cc, out_cls_mlo, out_match, _, _ = model_train(images_cc, images_mlo)
                        l_cls_cc = loss_cls(out_cls_cc, labels_cls_cc)
                        l_cls_mlo = loss_cls(out_cls_mlo, labels_cls_mlo)
                        l_match = loss_match(out_match, labels_match.squeeze(1).long()) 
                        l_total = (alpha * l_cls_cc) + (beta * l_cls_mlo) + (gamma * l_match)

                    scaler.scale(l_total).backward()
                    scaler.step(optimizer)
                    scaler.update()

                # 4. 统计损失
                total_loss += l_total.item()
                total_cls_loss += (l_cls_cc.item() + l_cls_mlo.item()) / 2.0
                total_match_loss += l_match.item()

                if local_rank == 0:
                    pbar.set_postfix(**{'total_loss'    : total_loss / (i + 1), 
                                        'cls_loss'      : total_cls_loss / (i + 1),
                                        'match_loss'    : total_match_loss / (i + 1),
                                        'lr'            : optimizer.param_groups[0]['lr']})
                    pbar.update(1)
            
            if local_rank == 0:
                pbar.close()
                print('Finish Train')
                print('Start Validation')
                pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

            # --- 开始验证 ---
            model_train.eval()
            for i, batch in enumerate(gen_val):
                if i >= epoch_step_val:
                    break

                # 1. 获取数据
                inputs, labels = batch
                images_cc, images_mlo = inputs
                labels_cls_cc, labels_cls_mlo, labels_match = labels
                
                if Cuda:
                    images_cc       = images_cc.cuda(local_rank)
                    images_mlo      = images_mlo.cuda(local_rank)
                    labels_cls_cc   = labels_cls_cc.cuda(local_rank)
                    labels_cls_mlo  = labels_cls_mlo.cuda(local_rank)
                    labels_match    = labels_match.cuda(local_rank)

                # 2. 前向传播 (不计算梯度)
                with torch.no_grad():
                    out_cls_cc, out_cls_mlo, out_match, _, _ = model_train(images_cc, images_mlo)
                    
                    l_cls_cc = loss_cls(out_cls_cc, labels_cls_cc)
                    l_cls_mlo = loss_cls(out_cls_mlo, labels_cls_mlo)
                    l_match = loss_match(out_match, labels_match.squeeze(1).long()) 
                    l_total = (alpha * l_cls_cc) + (beta * l_cls_mlo) + (gamma * l_match)

                # 3. 统计损失
                val_loss += l_total.item()
                val_cls_loss += (l_cls_cc.item() + l_cls_mlo.item()) / 2.0
                val_match_loss += l_match.item()
                
                if local_rank == 0:
                    pbar.set_postfix(**{'val_loss'      : val_loss / (i + 1),
                                        'val_cls'       : val_cls_loss / (i + 1),
                                        'val_match'     : val_match_loss / (i + 1),
                                        'lr'            : optimizer.param_groups[0]['lr']})
                    pbar.update(1)
            
            # --- 记录和保存 ---
            if local_rank == 0:
                pbar.close()
                print('Finish Validation')
                loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
                print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
                print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
                
                if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
                    torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if local_rank == 0:
            loss_history.writer.close()
