# #!/usr/bin/env python3
# """
# val.py - 使用 best_model.pth 验证 val 数据集准确率
# """
#
# import torch
# from torch.utils.data import DataLoader
# from config import (
#     DEVICE, BATCH_SIZE, TARGET_DATA_ROOT, USE_LIGHT_FILTER, LIGHT_SCENARIO
# )
# from dataset import OuluCASIADataset
# from model import DualBackboneDualStream  # 使用最优模型，可根据需要修改导入
#
# # --------------------- 加载数据集 ---------------------
# val_dataset = OuluCASIADataset(split='val', light_scenario=None)   # 默认全部光照
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
#
# print(f"✅ 验证集样本数：{len(val_dataset)}")
#
# # --------------------- 加载模型 ---------------------
# model = DualBackboneDualStream().to(DEVICE)
# checkpoint = torch.load('output/best_model.pth', map_location=DEVICE)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
# print(f"✅ 模型加载成功 (验证精度 {checkpoint.get('best_val_acc', 'N/A')})")
#
# # --------------------- 验证循环 ---------------------
# correct = 0
# total = 0
# with torch.no_grad():
#     for vis_x, ir_x, labels in val_loader:
#         vis_x, ir_x, labels = vis_x.to(DEVICE), ir_x.to(DEVICE), labels.to(DEVICE)
#         outputs, _ = model(vis_x, ir_x)          # 注意模型返回元组
#         _, preds = torch.max(outputs, 1)
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)
#
# acc = correct / total * 100
# print(f"🎯 验证集准确率：{acc:.2f}% ({correct}/{total})")


#!/usr/bin/env python3
"""
model_test.py - 使用 best_model.pth 验证 val 数据集准确率（分光照和整体）
"""

import torch
from torch.utils.data import DataLoader
from config import (
    DEVICE, BATCH_SIZE, TARGET_DATA_ROOT, USE_LIGHT_FILTER, LIGHT_SCENARIO
)
from dataset import OuluCASIADataset
from model import DualBackboneDualStream   # 根据实际情况修改导入

# --------------------- 加载模型 ---------------------
model = DualBackboneDualStream().to(DEVICE)
checkpoint = torch.load('output/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ 模型加载成功 (训练时最佳验证精度: {checkpoint.get('best_val_acc', 'N/A')})")

# --------------------- 定义测试函数 ---------------------
def test_accuracy(light_scenario=None):
    """测试指定光照场景（或全部）的准确率"""
    dataset = OuluCASIADataset(split='val', light_scenario=light_scenario)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    correct = 0
    total = 0
    with torch.no_grad():
        for vis_x, ir_x, labels in loader:
            vis_x, ir_x, labels = vis_x.to(DEVICE), ir_x.to(DEVICE), labels.to(DEVICE)
            outputs, _ = model(vis_x, ir_x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total * 100 if total > 0 else 0.0
    return acc, correct, total


# --------------------- 整体测试 ---------------------
print("\n========== 整体验证集准确率 ==========")
acc_all, corr_all, tot_all = test_accuracy(light_scenario=None)
print(f"🔸 整体   : {acc_all:.2f}% ({corr_all}/{tot_all})")