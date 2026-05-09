# #!/usr/bin/env python3
# # train_temporal.py – 支持变长序列的时序训练脚本（温和启动版）
#
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import LambdaLR
# import math
# from torchvision import transforms
#
# from config import (
#     DEVICE, BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS,
#     OUTPUT_DIR, EARLY_STOPPING_PATIENCE, LIGHT_SCENARIO,
#     NUM_CLASSES
# )
# from temporal_dataset import TemporalOuluCASIADataset
# from temporal_model import TemporalDualStream
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# TEMPORAL_DATA_ROOT = "data/temporal_data"
#
# class TemporalTrainer:
#     def __init__(self, model, light_scenario):
#         self.model = model.to(DEVICE)
#         self.light_scenario = light_scenario
#         self.criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
#
#         # 优化器（统一学习率，LR 在 config.py 中已设为 1e-4）
#         self.optimizer = optim.AdamW(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
#
#         # 学习率调度：预热 15 个 epoch，然后余弦退火
#         warmup_epochs = 15
#         total_epochs = EPOCHS
#         min_lr_ratio = 0.02
#         def lr_lambda(epoch):
#             if epoch < warmup_epochs:
#                 return (epoch + 1) / warmup_epochs
#             progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
#             cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
#             return max(cosine_decay, min_lr_ratio)
#         self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
#
#         # 暂时关闭所有数据增强
#         self.train_dataset = TemporalOuluCASIADataset(
#             json_path=os.path.join(TEMPORAL_DATA_ROOT, 'train_sequences.json'),
#             data_root=TEMPORAL_DATA_ROOT,
#             rgb_color_jitter=None,
#             rotation_deg=0,
#             hflip_prob=0.0
#         )
#         self.val_dataset = TemporalOuluCASIADataset(
#             json_path=os.path.join(TEMPORAL_DATA_ROOT, 'val_sequences.json'),
#             data_root=TEMPORAL_DATA_ROOT,
#             rgb_color_jitter=None,
#             rotation_deg=0,
#             hflip_prob=0.0
#         )
#
#         self.train_loader = DataLoader(
#             self.train_dataset, batch_size=BATCH_SIZE, shuffle=True,
#             num_workers=0, collate_fn=TemporalOuluCASIADataset.collate_fn
#         )
#         self.val_loader = DataLoader(
#             self.val_dataset, batch_size=BATCH_SIZE, shuffle=False,
#             num_workers=0, collate_fn=TemporalOuluCASIADataset.collate_fn
#         )
#
#         print(f"📊 训练集序列数：{len(self.train_dataset)}")
#         print(f"📊 验证集序列数：{len(self.val_dataset)}")
#
#         os.makedirs(OUTPUT_DIR, exist_ok=True)
#         self.best_val_acc = 0.0
#         self.early_stop_count = 0
#         self.train_losses, self.train_accs = [], []
#         self.val_losses, self.val_accs = [], []
#
#     def train_one_epoch(self, epoch):
#         self.model.train()
#         total_loss = 0.0
#         correct = total = 0
#
#         for batch_idx, (vis_seq, ir_seq, labels, lengths) in enumerate(self.train_loader):
#             vis_seq = vis_seq.to(DEVICE)
#             ir_seq = ir_seq.to(DEVICE)
#             labels = labels.to(DEVICE)
#             lengths = lengths.to(DEVICE)
#
#             outputs = self.model(vis_seq, ir_seq, lengths)
#             if isinstance(outputs, tuple):
#                 outputs = outputs[0]
#
#             loss = self.criterion(outputs, labels)
#
#             self.optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#             self.optimizer.step()
#
#             total_loss += loss.item()
#             _, preds = torch.max(outputs, 1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#
#             if batch_idx % 10 == 0:
#                 batch_acc = correct / total if total > 0 else 0.0
#                 print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(self.train_loader)}] "
#                       f"Loss: {loss.item():.4f} Acc: {batch_acc:.4f}")
#
#         avg_loss = total_loss / len(self.train_loader)
#         avg_acc = correct / total if total > 0 else 0.0
#         print(f"📈 Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}")
#         self.train_losses.append(avg_loss)
#         self.train_accs.append(avg_acc)
#         lr = self.optimizer.param_groups[0]['lr']
#         print(f"🔧 当前学习率: {lr:.6f}")
#         return avg_loss, avg_acc
#
#     def validate(self):
#         self.model.eval()
#         total_loss = 0.0
#         correct = total = 0
#         with torch.no_grad():
#             for vis_seq, ir_seq, labels, lengths in self.val_loader:
#                 vis_seq = vis_seq.to(DEVICE)
#                 ir_seq = ir_seq.to(DEVICE)
#                 labels = labels.to(DEVICE)
#                 lengths = lengths.to(DEVICE)
#
#                 outputs = self.model(vis_seq, ir_seq, lengths)
#                 if isinstance(outputs, tuple):
#                     outputs = outputs[0]
#
#                 loss = self.criterion(outputs, labels)
#                 total_loss += loss.item()
#                 _, preds = torch.max(outputs, 1)
#                 correct += (preds == labels).sum().item()
#                 total += labels.size(0)
#
#         avg_loss = total_loss / len(self.val_loader)
#         avg_acc = correct / total if total > 0 else 0.0
#         print(f"🔍 Val Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}")
#         self.val_losses.append(avg_loss)
#         self.val_accs.append(avg_acc)
#         return avg_loss, avg_acc
#
#     def plot_curves(self):
#         min_len = min(len(self.train_losses), len(self.val_losses))
#         epochs = range(1, min_len + 1)
#         plt.figure(figsize=(12, 4))
#         plt.subplot(1, 2, 1)
#         plt.plot(epochs, self.train_losses[:min_len], 'b-', label='Train Loss')
#         plt.plot(epochs, self.val_losses[:min_len], 'r-', label='Val Loss')
#         plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
#         plt.title('Temporal Model Loss')
#         plt.subplot(1, 2, 2)
#         plt.plot(epochs, self.train_accs[:min_len], 'b-', label='Train Acc')
#         plt.plot(epochs, self.val_accs[:min_len], 'r-', label='Val Acc')
#         plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
#         plt.title('Temporal Model Accuracy')
#         plt.tight_layout()
#         save_path = os.path.join(OUTPUT_DIR, 'training_curves_temporal.png')
#         plt.savefig(save_path, dpi=300)
#         plt.close()
#         print(f"📊 训练曲线已保存至 {save_path}")
#
#     def run(self):
#         print(f"\n🚀 开始时序训练（验证集独立，支持变长序列）...")
#         for epoch in range(EPOCHS):
#             self.train_one_epoch(epoch)
#             val_loss, val_acc = self.validate()
#             self.scheduler.step()
#
#             if val_acc > self.best_val_acc:
#                 self.best_val_acc = val_acc
#                 save_path = os.path.join(OUTPUT_DIR, 'best_temporal_model.pth')
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': self.model.state_dict(),
#                     'optimizer_state_dict': self.optimizer.state_dict(),
#                     'best_val_acc': self.best_val_acc,
#                 }, save_path)
#                 print(f"✅ 保存最优模型！当前验证准确率：{self.best_val_acc:.4f}")
#                 self.early_stop_count = 0
#             else:
#                 self.early_stop_count += 1
#                 if self.early_stop_count >= EARLY_STOPPING_PATIENCE:
#                     print(f"⚠️ 早停触发，终止训练")
#                     break
#
#         best_path = os.path.join(OUTPUT_DIR, 'best_temporal_model.pth')
#         if os.path.exists(best_path):
#             checkpoint = torch.load(best_path)
#             self.model.load_state_dict(checkpoint['model_state_dict'])
#         final_loss, final_acc = self.validate()
#         print(f"\n🎉 训练完成！最佳验证准确率：{self.best_val_acc:.4f}，最终：{final_acc:.4f}")
#         self.plot_curves()
#
#
# if __name__ == '__main__':
#     model = TemporalDualStream()
#     trainer = TemporalTrainer(model, light_scenario=LIGHT_SCENARIO)
#     trainer.run()
#!/usr/bin/env python3
# train_temporal.py – 支持变长序列的时序训练脚本（Transformer 优化版）


#!/usr/bin/env python3
# train_temporal.py – 稳定 LSTM 时序训练脚本

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import math
from torchvision import transforms

from config import (
    DEVICE, BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS,
    OUTPUT_DIR, EARLY_STOPPING_PATIENCE, LIGHT_SCENARIO,
    NUM_CLASSES
)
from temporal_dataset import TemporalOuluCASIADataset
from temporal_model import TemporalDualStream

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

TEMPORAL_DATA_ROOT = "data/temporal_data"

class TemporalTrainer:
    def __init__(self, model, light_scenario):
        self.model = model.to(DEVICE)
        self.light_scenario = light_scenario
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

        # 优化器（统一低学习率）
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        # 学习率调度：预热 15 个 epoch，然后余弦退火
        warmup_epochs = 15
        total_epochs = EPOCHS
        min_lr_ratio = 0.02
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return max(cosine_decay, min_lr_ratio)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        # 数据集：完全关闭数据增强，以稳定训练
        # self.train_dataset = TemporalOuluCASIADataset(
        #     json_path=os.path.join(TEMPORAL_DATA_ROOT, 'train_sequences.json'),
        #     data_root=TEMPORAL_DATA_ROOT,
        #     rgb_color_jitter=None,
        #     rotation_deg=0,
        #     hflip_prob=0.0
        # )
        color_jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05)
        self.train_dataset = TemporalOuluCASIADataset(
            json_path=os.path.join(TEMPORAL_DATA_ROOT, 'train_sequences.json'),
            data_root=TEMPORAL_DATA_ROOT,
            rgb_color_jitter=color_jitter,  # 暂不开启颜色抖动
            rotation_deg=5,  # 轻微旋转
            hflip_prob=0.5  # 水平翻转
        )
        self.val_dataset = TemporalOuluCASIADataset(
            json_path=os.path.join(TEMPORAL_DATA_ROOT, 'val_sequences.json'),
            data_root=TEMPORAL_DATA_ROOT,
            rgb_color_jitter=None,
            rotation_deg=0,
            hflip_prob=0.0
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, collate_fn=TemporalOuluCASIADataset.collate_fn
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, collate_fn=TemporalOuluCASIADataset.collate_fn
        )

        print(f"📊 训练集序列数：{len(self.train_dataset)}")
        print(f"📊 验证集序列数：{len(self.val_dataset)}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.best_val_acc = 0.0
        self.early_stop_count = 0
        self.train_losses, self.train_accs = [], []
        self.val_losses, self.val_accs = [], []

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = total = 0

        for batch_idx, (vis_seq, ir_seq, labels, lengths) in enumerate(self.train_loader):
            vis_seq = vis_seq.to(DEVICE)
            ir_seq = ir_seq.to(DEVICE)
            labels = labels.to(DEVICE)
            lengths = lengths.to(DEVICE)

            outputs = self.model(vis_seq, ir_seq, lengths)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            # 稍微加强梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 10 == 0:
                batch_acc = correct / total if total > 0 else 0.0
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f} Acc: {batch_acc:.4f}")

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = correct / total if total > 0 else 0.0
        print(f"📈 Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}")
        self.train_losses.append(avg_loss)
        self.train_accs.append(avg_acc)
        lr = self.optimizer.param_groups[0]['lr']
        print(f"🔧 当前学习率: {lr:.6f}")
        return avg_loss, avg_acc

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for vis_seq, ir_seq, labels, lengths in self.val_loader:
                vis_seq = vis_seq.to(DEVICE)
                ir_seq = ir_seq.to(DEVICE)
                labels = labels.to(DEVICE)
                lengths = lengths.to(DEVICE)

                outputs = self.model(vis_seq, ir_seq, lengths)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        avg_acc = correct / total if total > 0 else 0.0
        print(f"🔍 Val Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}")
        self.val_losses.append(avg_loss)
        self.val_accs.append(avg_acc)
        return avg_loss, avg_acc

    def plot_curves(self):
        min_len = min(len(self.train_losses), len(self.val_losses))
        epochs = range(1, min_len + 1)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses[:min_len], 'b-', label='Train Loss')
        plt.plot(epochs, self.val_losses[:min_len], 'r-', label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
        plt.title('Temporal Model Loss')
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accs[:min_len], 'b-', label='Train Acc')
        plt.plot(epochs, self.val_accs[:min_len], 'r-', label='Val Acc')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
        plt.title('Temporal Model Accuracy')
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, 'training_curves_temporal.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📊 训练曲线已保存至 {save_path}")

    def run(self):
        print(f"\n🚀 开始时序训练（LSTM 稳定版）...")
        for epoch in range(EPOCHS):
            self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_path = os.path.join(OUTPUT_DIR, 'best_temporal_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                }, save_path)
                print(f"✅ 保存最优模型！当前验证准确率：{self.best_val_acc:.4f}")
                self.early_stop_count = 0
            else:
                self.early_stop_count += 1
                if self.early_stop_count >= EARLY_STOPPING_PATIENCE:
                    print(f"⚠️ 早停触发，终止训练")
                    break

        best_path = os.path.join(OUTPUT_DIR, 'best_temporal_model.pth')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        final_loss, final_acc = self.validate()
        print(f"\n🎉 训练完成！最佳验证准确率：{self.best_val_acc:.4f}，最终：{final_acc:.4f}")
        self.plot_curves()


if __name__ == '__main__':
    model = TemporalDualStream()
    trainer = TemporalTrainer(model, light_scenario=LIGHT_SCENARIO)
    trainer.run()