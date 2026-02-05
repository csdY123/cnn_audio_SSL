# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time

from dataset_sim import DynamicRoomSimulator 
from model import RawAudioSSLNet


# ==========================================
# 2. 训练主循环
# ==========================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import os

from dataset_sim import DynamicRoomSimulator

def main():
    AUDIO_DIR = "./speech_data/LibriSpeech/dev-clean" # 确保路径正确
    BATCH_SIZE = 256
    LR = 0.001
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR = "saved_2_after60stpes_addrealnoise"
    
    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 创建日志文件
    log_file = os.path.join(SAVE_DIR, "training_log.txt")
    log_f = open(log_file, "a")

    # 1. 加载数据
    dataset = DynamicRoomSimulator(audio_source_dir=AUDIO_DIR, sample_length=2048, epoch_length=20000)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 2. 模型
    model = RawAudioSSLNet().to(DEVICE)
    model.load_state_dict(torch.load("/mnt/chensenda/codes/sound/cnn_audio_SSL/saved_2_after60stpes_addrealnoise/ssl_model_reg_5.pth"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # 学习率调度: 从1e-3线性下降到1e-4，在0-50 epoch内完成
    LR_START = 1e-3
    LR_END = 1e-4
    LR_DECAY_EPOCHS = 100
    
    def lr_lambda(epoch):
        if epoch < LR_DECAY_EPOCHS:
            # 线性插值: lr = LR_START - (LR_START - LR_END) * epoch / LR_DECAY_EPOCHS
            return 1.0 - (1.0 - LR_END / LR_START) * epoch / LR_DECAY_EPOCHS
        else:
            return LR_END / LR_START
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # --- 改进 3: 使用 MSE Loss ---
    criterion = nn.MSELoss()

    print("开始训练 (Regression Mode: Sin/Cos)...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, (inputs, target_degrees) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            
            # --- 改进 4: 将角度 Label 转换为 Sin/Cos ---
            # target_degrees shape: [Batch]
            deg_rads = torch.deg2rad(target_degrees.float()).to(DEVICE)
            
            target_sin = torch.sin(deg_rads)
            target_cos = torch.cos(deg_rads)
            
            # 拼接成 [Batch, 2]
            target_vec = torch.stack((target_sin, target_cos), dim=1)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs) # outputs: [Batch, 2]
            
            # 计算 Loss (让预测的向量去逼近真实的 Sin/Cos)
            loss = criterion(outputs, target_vec)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # --- 计算准确率 (还原回角度) ---
            with torch.no_grad():
                # 使用 atan2(sin, cos) 还原角度
                pred_sin = outputs[:, 0]
                pred_cos = outputs[:, 1]
                pred_rads = torch.atan2(pred_sin, pred_cos)
                pred_degs = torch.rad2deg(pred_rads)
                
                # 转换到 [0, 360) 区间
                pred_degs = (pred_degs + 360) % 360
                real_degs = target_degrees.to(DEVICE).float()
                
                # 计算误差
                diff = torch.abs(pred_degs - real_degs)
                diff = torch.min(diff, 360 - diff) # 解决 0 vs 360 问题
                
                # 统计误差 < 10度的
                correct += (diff <= 15).sum().item()
                total += inputs.size(0)

            if batch_idx % 10 == 0:
                print(f"\rEpoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}", end="")

        # 打印 Epoch 结果
        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        duration = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        log_msg = f"Epoch {epoch} | Time: {duration:.1f}s | MSE Loss: {avg_loss:.4f} | Acc (±15°): {acc:.2f}% | LR: {current_lr:.6f}"
        print(f"\n{log_msg}")
        log_f.write(log_msg + "\n")
        log_f.flush()
        
        # 更新学习率
        scheduler.step()
        
        # 只有当准确率看起来正常了才保存 (比如 > 10%)
        if epoch % 5 == 0:
             torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"ssl_model_reg_{epoch}.pth"))
    
    log_f.close()
    print(f"训练完成，日志已保存到: {log_file}")

if __name__ == "__main__":
    main()