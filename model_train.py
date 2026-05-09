#
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import (
    DEVICE, BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS,
    OUTPUT_DIR, EARLY_STOPPING_PATIENCE, LIGHT_SCENARIO
)
from dataset import OuluCASIADataset
from model import DualBackboneDualStream

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

TRAIN_VAL_SPLIT_RATIO = 0.8

class ScenarioTrainer:
    def __init__(self, model, light_scenario):
        self.model = model.to(DEVICE)
        self.light_scenario = light_scenario  # 可能为 None, 'weak', 'strong', 'dark'
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 微分学习率配置
        vis_backbone_params = list(self.model.vis_backbone.parameters())
        ir_backbone_params = list(self.model.ir_backbone.parameters())
        fusion_params = list(self.model.fusion.parameters())
        head_params = (list(self.model.fusion_head.parameters()) +
                       list(self.model.vis_head.parameters()) +
                       list(self.model.ir_head.parameters()))

        optimizer_params = [
            {'params': vis_backbone_params, 'lr': LR * 0.1},
            {'params': ir_backbone_params, 'lr': LR * 0.1},
            {'params': fusion_params, 'lr': LR},
            {'params': head_params, 'lr': LR},
        ]
        self.optimizer = optim.AdamW(optimizer_params, weight_decay=WEIGHT_DECAY)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)

        # 加载数据集（light_scenario 为 None 时返回全部样本）
        full_train_dataset = OuluCASIADataset(split='train', light_scenario=light_scenario)
        scenario_display = light_scenario if light_scenario is not None else "全部(增强合并)"
        print(f"📊 加载[{scenario_display}]场景训练集：共{len(full_train_dataset)}条样本")

        train_size = int(TRAIN_VAL_SPLIT_RATIO * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_subset, val_subset = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"✅ 数据集拆分完成：训练子集{len(train_subset)}条 | 验证子集{len(val_subset)}条")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.best_val_acc = 0.0
        self.early_stop_count = 0

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = total = 0

        for batch_idx, (vis_x, ir_x, labels) in enumerate(self.train_loader):
            vis_x, ir_x, labels = vis_x.to(DEVICE), ir_x.to(DEVICE), labels.to(DEVICE)
            outputs = self.model(vis_x, ir_x)

            # ✅ 修改点：取输出的第一个元素（分类logits）
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}] | Batch [{batch_idx}/{len(self.train_loader)}] | "
                      f"Loss: {loss.item():.4f} | Acc: {correct / total:.4f}")

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = correct / total
        self.train_losses.append(avg_loss)
        self.train_accs.append(avg_acc)
        print(f"📈 Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f}")
        print(f"🔧 当前学习率：{self.optimizer.param_groups[0]['lr']:.6f}")
        return avg_loss, avg_acc

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for vis_x, ir_x, labels in self.val_loader:
                vis_x, ir_x, labels = vis_x.to(DEVICE), ir_x.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(vis_x, ir_x)

                # ✅ 修改点：取输出的第一个元素
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                _, preds = torch.max(logits, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = correct / total
        print(f"🔍 Val Loss: {avg_loss:.4f} | Val Acc: {avg_acc:.4f}")
        self.val_losses.append(avg_loss)
        self.val_accs.append(avg_acc)
        return avg_loss, avg_acc

    def plot_curves(self):
        min_len = min(len(self.train_losses), len(self.val_losses))
        epochs = range(1, min_len + 1)
        train_losses = self.train_losses[:min_len]
        train_accs = self.train_accs[:min_len]
        val_losses = self.val_losses[:min_len]
        val_accs = self.val_accs[:min_len]

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, val_losses, 'r-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        title_str = f'[{self.light_scenario if self.light_scenario else "Merged"}] Loss Curves'
        plt.title(title_str)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, 'b-', label='Train Acc')
        plt.plot(epochs, val_accs, 'r-', label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f'[{self.light_scenario if self.light_scenario else "Merged"}] Accuracy Curves')

        plt.tight_layout()
        if self.light_scenario is None:
            save_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
        else:
            save_path = os.path.join(OUTPUT_DIR, f'{self.light_scenario}_training_curves.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📊 训练曲线已保存：{save_path}")

    def run(self):
        scenario_display = self.light_scenario if self.light_scenario else "全部(增强合并)"
        print(f"\n🚀 开始[{scenario_display}]场景训练（训练集8:2拆分，验证集为优化目标）...")
        for epoch in range(EPOCHS):
            self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()
            self.scheduler.step(val_loss)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                if self.light_scenario is None:
                    save_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
                else:
                    save_path = os.path.join(OUTPUT_DIR, f'{self.light_scenario}_best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'light_scenario': self.light_scenario
                }, save_path)
                print(f"✅ 保存最优模型！当前最高验证精度：{self.best_val_acc:.4f}")
                self.early_stop_count = 0
            else:
                self.early_stop_count += 1
                if self.early_stop_count >= EARLY_STOPPING_PATIENCE:
                    print(f"⚠️ 早停触发（{EARLY_STOPPING_PATIENCE}轮无提升），终止训练")
                    break

        # 加载最优模型并最终验证
        if self.light_scenario is None:
            model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
        else:
            model_path = os.path.join(OUTPUT_DIR, f'{self.light_scenario}_best_model.pth')
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        final_val_loss, final_val_acc = self.validate()
        print(f"\n🎉 [{scenario_display}] 训练完成！")
        print(f"📌 最优验证精度：{checkpoint['best_val_acc']:.4f} | 最终验证精度：{final_val_acc:.4f}")
        self.plot_curves()


if __name__ == '__main__':
    model = DualBackboneDualStream()
    # 直接使用 config 中的 LIGHT_SCENARIO（可能为 None 或具体光照）
    trainer = ScenarioTrainer(model, light_scenario=LIGHT_SCENARIO)
    trainer.run()
