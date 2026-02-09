# train_ddp.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np
import time
import os
import argparse

from dataset_sim import DynamicRoomSimulator
from model import RawAudioSSLNet

def get_args():
    parser = argparse.ArgumentParser()
    # torchrun 会自动传递 local_rank，但为了兼容性还是保留
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def main():
    args = get_args()
    
    # ==========================================
    # 1. DDP 初始化设置
    # ==========================================
    # 检查环境变量，判断是否通过 torchrun 启动
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        use_ddp = True
    else:
        # 单卡模式 / CPU模式
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_ddp = False
        print("[Info] Running in Single GPU/CPU mode.")

    if is_main_process():
        print(f"[Info] Training with {world_size} GPU(s).")

    # ==========================================
    # 2. 配置参数
    # ==========================================
    AUDIO_DIR = "./speech_data/LibriSpeech/dev-clean"
    # 注意：DDP模式下，BATCH_SIZE 是指"每张卡"的大小
    # 如果你有8张卡，总 BatchSize = 256 * 8 = 2048
    BATCH_SIZE = 128 
    LR = 0.0005  # 学习率通常随总 BatchSize 线性缩放
    EPOCHS = 2000
    SAVE_DIR = "saved_middle_ddp_after_345_GPU3_after500steps_addrealnoise"
    
    if is_main_process():
        os.makedirs(SAVE_DIR, exist_ok=True)
        log_file = os.path.join(SAVE_DIR, "training_log.txt")
        log_f = open(log_file, "a")
    else:
        log_f = None

    # ==========================================
    # 3. 加载数据 (关键修改: Sampler)
    # ==========================================
    dataset = DynamicRoomSimulator(audio_source_dir=AUDIO_DIR, sample_length=2048, epoch_length=40960)
    
    if use_ddp:
        # DDP 必须使用 DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False # Sampler 已经负责 shuffle 了，DataLoader 里必须设为 False
    else:
        sampler = None
        shuffle = True

    # num_workers 建议设为 CPU 核心数 / GPU 数量，或者设为 4-8
    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=shuffle, 
        sampler=sampler, 
        num_workers=4, # 8 GPU 意味着你需要更强的数据吞吐，建议调大 worker
        pin_memory=True
    )

    # ==========================================
    # 4. 模型加载 (关键修改: DDP Wrap)
    # ==========================================
    model = RawAudioSSLNet()
    
    # # 如果有预训练权重，建议在 wrap DDP 之前加载
    pretrained_path = "/mnt/chensenda/codes/sound/cnn_audio_SSL/saved_middle_ddp_after_345_GPU3/ssl_model_ddp_495.pth"
    if os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)

    model.to(device)

    if use_ddp:
        # 将 BatchNorm 转换为 SyncBatchNorm (多卡同步 BN 统计量，提升精度)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # 包装模型
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # 学习率调度
    LR_START = 1e-3  # 同样缩放起始学习率
    LR_END = 1e-4 
    LR_DECAY_EPOCHS = 2000
    
    def lr_lambda(epoch):
        if epoch < LR_DECAY_EPOCHS:
            return 1.0 - (1.0 - LR_END / LR_START) * epoch / LR_DECAY_EPOCHS
        else:
            return LR_END / LR_START
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.MSELoss()

    # ==========================================
    # Load checkpoint if exists (model, optimizer, scheduler, epoch)
    # ==========================================
    start_epoch = 0
    checkpoint_path = "/mnt/chensenda/codes/sound/cnn_audio_SSL/saved_middle_ddp_after_345_GPU3/checkpoint_latest.pth"
    if os.path.exists(checkpoint_path):
        if is_main_process():
            print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model_to_load = model.module if use_ddp else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Resume from next epoch
        start_epoch = checkpoint['epoch'] + 1
        
        if is_main_process():
            print(f"Resumed from epoch {checkpoint['epoch']}, starting epoch {start_epoch}")

    if is_main_process():
        print("开始训练 (Regression Mode: Sin/Cos)...")
    
    # ==========================================
    # 5. 训练循环
    # ==========================================
    for epoch in range(start_epoch, EPOCHS):
        # DDP 关键：每个 epoch 开始前必须设置 sampler 的 epoch，否则 shuffle 不生效
        if use_ddp:
            sampler.set_epoch(epoch)
            
        model.train()
        
        # 统计变量 (只在本地统计，最后打印时如果需要可以做 all_reduce，但通常只看 rank0 的也够了)
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, (inputs, target_degrees) in enumerate(train_loader):
            inputs = inputs.to(device)
            target_degrees = target_degrees.to(device)
            
            # Label 转换
            deg_rads = torch.deg2rad(target_degrees.float())
            target_sin = torch.sin(deg_rads)
            target_cos = torch.cos(deg_rads)
            target_vec = torch.stack((target_sin, target_cos), dim=1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target_vec)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            # 计算准确率
            with torch.no_grad():
                pred_sin = outputs[:, 0]
                pred_cos = outputs[:, 1]
                pred_rads = torch.atan2(pred_sin, pred_cos)
                pred_degs = torch.rad2deg(pred_rads)
                pred_degs = (pred_degs + 360) % 360
                
                diff = torch.abs(pred_degs - target_degrees.float())
                diff = torch.min(diff, 360 - diff)
                correct += (diff <= 15).sum().item()
                total += inputs.size(0)

            # 仅主进程打印进度
            if is_main_process() and batch_idx % 10 == 0:
                print(f"\rEpoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}", end="")

        # 更新 LR
        scheduler.step()

        # -----------------------------------------------------------
        # Aggregate statistics across all GPUs using all_reduce
        # -----------------------------------------------------------
        if use_ddp:
            # Create tensors for reduction
            stats = torch.tensor([total_loss, correct, total], dtype=torch.float64, device=device)
            
            # Sum across all processes
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            
            # Extract aggregated values
            total_loss_all = stats[0].item()
            correct_all = stats[1].item()
            total_all = stats[2].item()
            
            # Calculate global average loss and accuracy
            # Note: len(train_loader) is the same on all ranks, so we multiply by world_size
            avg_loss = total_loss_all / (len(train_loader) * world_size)
            acc = 100 * correct_all / total_all if total_all > 0 else 0.0
        else:
            avg_loss = total_loss / len(train_loader)
            acc = 100 * correct / total if total > 0 else 0.0

        # 打印日志 (仅 Rank 0)
        if is_main_process():
            duration = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            log_msg = f"Epoch {epoch} | Time: {duration:.1f}s | MSE Loss: {avg_loss:.4f} | Acc (±15°): {acc:.2f}% | LR: {current_lr:.6f}"
            print(f"\n{log_msg}")
            
            if log_f:
                log_f.write(log_msg + "\n")
                log_f.flush()
            
            # 保存模型 (仅 Rank 0 保存，且保存 model.module)
            if epoch % 5 == 0:
                # 注意：DDP 包装的模型要用 .module 取出原始模型
                model_to_save = model.module if use_ddp else model
                
                # Save complete checkpoint with optimizer and scheduler state
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                }
                
                # Save periodic checkpoint
                torch.save(checkpoint, os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch}.pth"))
                
                # Save latest checkpoint for easy resume
                torch.save(checkpoint, os.path.join(SAVE_DIR, "checkpoint_latest.pth"))
                
                # Also save model-only file for inference (backward compatible)
                torch.save(model_to_save.state_dict(), os.path.join(SAVE_DIR, f"ssl_model_ddp_{epoch}.pth"))

    if is_main_process() and log_f:
        log_f.close()
        print(f"训练完成，日志已保存到: {SAVE_DIR}")

    # 销毁进程组
    if use_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()