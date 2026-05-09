#
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import math
from config import (
    DEVICE, BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS,
    OUTPUT_DIR, EARLY_STOPPING_PATIENCE, LIGHT_SCENARIO,IR_BACKBONE_LR_RATIO
)
from dataset import OuluCASIADataset
from model import DualBackboneDualStream

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Trainer:
    def __init__(self, model, light_scenario):
        self.model = model.to(DEVICE)
        self.light_scenario = light_scenario
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

        # ---------- 梯度平衡优化：调整各参数组学习率 ----------
        # VIS 主干：适度提高学习率，打破“冻结”假象
        vis_backbone_params = [p for p in self.model.vis_backbone.parameters() if p.requires_grad]
        # IR 主干：大幅降低学习率，防止主导训练
        ir_backbone_params = [p for p in self.model.ir_backbone.parameters() if p.requires_grad]
        # 融合模块整体参数
        fusion_params = list(self.model.fusion.parameters())
        # 融合头部参数
        head_params = (list(self.model.fusion_head.parameters()) +
                       list(self.model.vis_head.parameters()) +
                       list(self.model.ir_head.parameters()))
        # 自注意力单独分组，给予更高学习率
        self_attn_params = list(self.model.fusion.self_attn.parameters()) if hasattr(self.model.fusion, 'self_attn') else []

        optimizer_params = []
        if vis_backbone_params:
            optimizer_params.append({'params': vis_backbone_params, 'lr': LR * 0.5})      # 提升 VIS 学习率
        if ir_backbone_params:
            optimizer_params.append({'params': ir_backbone_params, 'lr': LR * 0.2 * IR_BACKBONE_LR_RATIO})       # 抑制 IR 学习率
        if self_attn_params:
            optimizer_params.append({'params': self_attn_params, 'lr': LR * 2.0})         # 自注意力 2× 学习率
        if fusion_params:
            # 融合模块中排除自注意力（已单独分组），避免重复
            other_fusion_params = [p for p in fusion_params if p not in set(self_attn_params)]
            if other_fusion_params:
                optimizer_params.append({'params': other_fusion_params, 'lr': LR})
        if head_params:
            optimizer_params.append({'params': head_params, 'lr': LR})

        self.optimizer = optim.AdamW(optimizer_params, weight_decay=WEIGHT_DECAY)

        # 学习率调度（预热 + 余弦退火，下限 0.02*LR）
        warmup_epochs = 8
        total_epochs = EPOCHS
        min_lr_ratio = 0.02

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return max(cosine_decay, min_lr_ratio)

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        # 数据集
        self.train_dataset = OuluCASIADataset(split='train', light_scenario=light_scenario)
        self.val_dataset = OuluCASIADataset(split='val', light_scenario=light_scenario)
        self.train_loader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        print(f"📊 训练集样本数：{len(self.train_dataset)}")
        print(f"📊 验证集样本数：{len(self.val_dataset)}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.best_val_acc = 0.0
        self.early_stop_count = 0
        self.train_losses, self.train_accs = [], []
        self.val_losses, self.val_accs = [], []

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = total = 0

        for batch_idx, (vis_x, ir_x, labels) in enumerate(self.train_loader):
            vis_x, ir_x, labels = vis_x.to(DEVICE), ir_x.to(DEVICE), labels.to(DEVICE)
            outputs, _ = self.model(vis_x, ir_x)
            loss = self.criterion(outputs, labels)


            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止 IR 梯度过大导致震荡
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 10 == 0:
                batch_acc = correct / total if total > 0 else 0.0
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(self.train_loader)}] | "
                      f"Loss: {loss.item():.4f} | Train Acc: {batch_acc:.4f}")

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = correct / total if total > 0 else 0.0
        print(f"📈 Epoch [{epoch+1}/{EPOCHS}] | Train Avg Loss: {avg_loss:.4f} | Train Avg Acc: {avg_acc:.4f}")
        self.train_losses.append(avg_loss)
        self.train_accs.append(avg_acc)
        # 打印融合层学习率（可能存在多个组）
        lr = self.optimizer.param_groups[-1]['lr']  # 假设最后一组是 head_params
        print(f"🔧 当前学习率（融合层）：{lr:.6f}")
        return avg_loss, avg_acc

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for vis_x, ir_x, labels in self.val_loader:
                vis_x, ir_x, labels = vis_x.to(DEVICE), ir_x.to(DEVICE), labels.to(DEVICE)
                outputs, _ = self.model(vis_x, ir_x)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = correct / total if total > 0 else 0.0
        print(f"🔍 Val Avg Loss: {avg_loss:.4f} | Val Accuracy: {avg_acc:.4f}")
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
        plt.title(f'[{self.light_scenario or "Merged"}] Loss Curves')
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accs[:min_len], 'b-', label='Train Acc')
        plt.plot(epochs, self.val_accs[:min_len], 'r-', label='Val Acc')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
        plt.title(f'[{self.light_scenario or "Merged"}] Accuracy Curves')
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, 'training_curves.png' if self.light_scenario is None else f'{self.light_scenario}_training_curves.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📊 训练曲线已保存：{save_path}")

    def run(self):
        scenario_display = self.light_scenario if self.light_scenario else "增强合并(全部)"
        print(f"\n🚀 开始 [{scenario_display}] 训练（验证集为后15人）...")
        for epoch in range(EPOCHS):
            self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_path = os.path.join(OUTPUT_DIR, 'best_model.pth' if self.light_scenario is None else f'{self.light_scenario}_best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'light_scenario': self.light_scenario
                }, save_path)
                print(f"✅ 保存最优模型！当前验证精度：{self.best_val_acc:.4f}")
                self.early_stop_count = 0
            else:
                self.early_stop_count += 1
                if self.early_stop_count >= EARLY_STOPPING_PATIENCE:
                    print(f"⚠️ 早停触发（{EARLY_STOPPING_PATIENCE}轮无提升），终止训练")
                    break

        # 最终验证
        model_path = os.path.join(OUTPUT_DIR, 'best_model.pth' if self.light_scenario is None else f'{self.light_scenario}_best_model.pth')
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        final_val_loss, final_val_acc = self.validate()
        print(f"\n🎉 [{scenario_display}] 训练完成！")
        print(f"📌 最优验证精度：{checkpoint['best_val_acc']:.4f} | 最终验证精度：{final_val_acc:.4f}")
        self.plot_curves()

if __name__ == '__main__':
    model = DualBackboneDualStream()
    trainer = Trainer(model, light_scenario=LIGHT_SCENARIO)
    trainer.run()
