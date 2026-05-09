import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from config import DEVICE, BATCH_SIZE
from dataset import OuluCASIADataset
from model import DualBackboneDualStream

# 情绪标签（与你的数据集一致）
EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

# 加载模型
model = DualBackboneDualStream().to(DEVICE)
checkpoint = torch.load('output/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 加载测试集（例如 val 集）
test_dataset = OuluCASIADataset(split='val')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for vis_x, ir_x, labels in test_loader:
        vis_x, ir_x = vis_x.to(DEVICE), ir_x.to(DEVICE)
        outputs, _ = model(vis_x, ir_x)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)

# 绘制
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTIONS, yticklabels=EMOTIONS)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# 输出分类报告
print(classification_report(all_labels, all_preds, target_names=EMOTIONS))