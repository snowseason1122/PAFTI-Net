import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from torch.utils.data import DataLoader, TensorDataset
import os

# ===================== 配置 =====================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
IN_FEATURES = 15
NUM_CLASSES = 3
BATCH_SIZE = 16  # 固定为之前最优batch
D_MODEL = 32

# 待测试学习率列表，可自行增删
LR_LIST = [0.1, 0.01, 0.001, 0.0001, 0.0005]


# ===================== 标签 =====================
def convert_label(raw):
    if raw <= 100:
        return 0
    elif raw <= 499:
        return 1
    else:
        return 2


# ===================== 数据加载 =====================
def load_data():
    data_folder = r"Data"

    df_train = pd.read_excel(os.path.join(data_folder, "train.xlsx"))
    df_val = pd.read_excel(os.path.join(data_folder, "valid.xlsx"))
    df_test = pd.read_excel(os.path.join(data_folder, "test.xlsx"))

    print(f"训练集: {len(df_train)} | 验证集: {len(df_val)} | 测试集: {len(df_test)}")

    feat_cols = [
        'co2_均值', 'co2_中位数', 'co2_最大值', 'co2_最小值', 'co2_标准差', 'co2_斜率',
        'temp_均值', 'temp_中位数', 'temp_最大值', 'temp_最小值', 'temp_标准差', 'temp_斜率',
        'hum_均值', 'hum_中位数', 'hum_最大值'
    ]

    def get_xy(df):
        X = df[feat_cols].values
        y = df["标签"].apply(convert_label).values
        X = np.nan_to_num(X)
        return X, y

    X_train, y_train = get_xy(df_train)
    X_val, y_val = get_xy(df_val)
    X_test, y_test = get_xy(df_test)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ===================== 模型 =====================
class AFFM(nn.Module):
    def __init__(self, dim=15):
        super().__init__()
        self.fc1 = nn.Linear(dim, 8)
        self.fc2 = nn.Linear(8, dim)

    def forward(self, x):
        w = torch.sigmoid(self.fc2(torch.tanh(self.fc1(x))))
        return x * w


class MCP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.c1 = nn.Conv1d(in_dim, 16, 1)
        self.c3 = nn.Conv1d(in_dim, 16, 3, padding=1)
        self.c5 = nn.Conv1d(in_dim, 16, 5, padding=2)
        self.p = nn.Conv1d(48, out_dim, 1)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.cat([self.c1(x), self.c3(x), self.c5(x)], dim=1)
        return self.bn(self.p(x)).transpose(1, 2)


class CFM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(15, dim)
        self.pe = nn.Parameter(torch.randn(1, 1, dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 2, dim * 2, batch_first=True), 1)

    def forward(self, x):
        return self.encoder(self.proj(x) + self.pe)


class FIP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(15, 64)
        self.fc2 = nn.Linear(64, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        B, T, C = x.shape
        x = F.relu(self.fc1(x.reshape(B * T, C)))
        return self.bn(self.fc2(x)).reshape(B, T, -1)


class PAFTI_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.affm = AFFM(15)
        self.mcp = MCP(15, D_MODEL)
        self.cfm = CFM(D_MODEL)
        self.fip = FIP(D_MODEL)
        self.fc = nn.Sequential(nn.Linear(D_MODEL * 3, D_MODEL), nn.ReLU())
        self.head = nn.Linear(D_MODEL, NUM_CLASSES)

    def forward(self, x):
        x = self.affm(x)
        a = self.mcp(x)
        b = self.cfm(x)
        c = self.fip(x)
        out = a + b + c
        out = out.mean(dim=1)
        return self.head(out)


# ===================== 单学习率训练函数 =====================
def train_with_lr(lr):
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
                              BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
                            BATCH_SIZE, shuffle=False)

    model = PAFTI_Net().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证集评估
        model.eval()
        v_pred, v_true = [], []
        with torch.no_grad():
            for x, y in val_loader:
                v_pred.extend(model(x.to(DEVICE)).argmax(1).cpu().numpy())
                v_true.extend(y.numpy())
        val_acc = accuracy_score(v_true, v_pred)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"LR={lr:.5f} | Epoch {epoch + 1} | Loss: {total_loss / len(train_loader):.3f} | Val Acc: {val_acc:.4f}")

    return best_val_acc


# ===================== 主实验：遍历学习率 + 绘图 + 保存Excel =====================
def run_lr_experiment():
    print("\n========== 开始实验：最优 Learning Rate ==========\n")
    lr_results = {}

    # 遍历所有学习率
    for lr in LR_LIST:
        current_best_acc = train_with_lr(lr)
        lr_results[lr] = current_best_acc
        print(f"\n✅ 学习率 {lr:.5f} 最优验证集准确率: {current_best_acc:.4f}\n")

    # 筛选最优学习率
    best_lr = max(lr_results, key=lr_results.get)
    best_acc = lr_results[best_lr]
    print("=" * 60)
    print(f"🏆 最优学习率 = {best_lr:.5f}，对应最高准确率 = {best_acc:.4f}")
    print("=" * 60)

    # ===================== 新增：保存实验结果到Excel =====================
    # 整理数据
    result_data = []
    for lr, acc in lr_results.items():
        result_data.append({
            "学习率(Learning Rate)": lr,
            "最优验证集准确率(Best Val Accuracy)": round(acc, 4),
            "是否最优": "是" if lr == best_lr else "否"
        })

    # 转为DataFrame
    df_result = pd.DataFrame(result_data)

    # 保存到Excel
    excel_path = "lr学习率实验结果.xlsx"
    df_result.to_excel(excel_path, index=False, sheet_name="实验结果")
    print(f"\n📊 实验结果已保存到Excel：{excel_path}")

    # ===================== 绘制柱状图 =====================
    plt.figure(figsize=(9, 5))
    x_ticks = [str(lr) for lr in lr_results.keys()]
    acc_values = list(lr_results.values())
    bars = plt.bar(x_ticks, acc_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

    plt.title('Learning Rate 对比实验结果（最优验证集准确率）', fontsize=14)
    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.05)

    # 柱子顶部标注数值
    for bar, acc in zip(bars, acc_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig("lr对比图.png", dpi=300)
    plt.close()
    print("\n📊 学习率对比柱状图已保存：lr对比图.png")


if __name__ == "__main__":
    run_lr_experiment()