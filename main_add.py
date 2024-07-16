import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import scipy.signal as signal
import torch.nn.functional as F

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=271, Samples=281, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False),  # Conv2D
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, padding='valid', bias=False),  # DepthwiseConv2D
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False),  # SeparableConv2D
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )

        # プーリング後の形状に基づいて正しい入力サイズを計算
        self.flatten_size = F2 * (Samples // 4 // 8)  # まず4で割って次に8で割る

        self.dense = nn.Linear(self.flatten_size, nb_classes)  # 修正した入力サイズ
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 1, 271, 281)  # 入力形状の調整
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        x = self.softmax(x)
        return x


fs = 200

# 帯域通過フィルタの設定
lowcut = 0.5
highcut = 50.0
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist
b, a = signal.butter(2, [low, high], btype='band')  # フィルターの次数を2に増やしました


# データの前処理関数
def preprocess_eeg(data):
    # データをフィルタリング
    filtered_data = signal.lfilter(b, a, data, axis=-1)

    # ベースライン補正（データ全体の平均を引く）
    baseline = np.mean(filtered_data, axis=-1, keepdims=True)
    corrected_data = filtered_data - baseline

    # データを正規化 (標準化)
    mean = np.mean(corrected_data, axis=-1, keepdims=True)
    std = np.std(corrected_data, axis=-1, keepdims=True)
    normalized_data = (corrected_data - mean) / std
    return normalized_data


# データの読み込み
x_train = torch.load('C:/Users/yabe0/PycharmProjects/MEG/data-001/train_X.pt')
y_train = torch.load('C:/Users/yabe0/PycharmProjects/MEG/data-001/train_y.pt')
x_val = torch.load('C:/Users/yabe0/PycharmProjects/MEG/data-001/val_X.pt')
y_val = torch.load('C:/Users/yabe0/PycharmProjects/MEG/data-001/val_y.pt')
x_test = torch.load('C:/Users/yabe0/PycharmProjects/MEG/data-001/test_X.pt')


# データセットクラス
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data=None):
        self.x_data = x_data.to(torch.float32)
        self.y_data = y_data

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        if self.y_data is not None:
            return self.x_data[idx], self.y_data[idx]
        else:
            return self.x_data[idx]


train_data = CustomDataset(x_train, y_train)
val_data = CustomDataset(x_val, y_val)
test_data = CustomDataset(x_test)

# データローダー
batch_size = 128
dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
dataloader_val = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
dataloader_test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


# 前処理をバッチごとに行う関数
def preprocess_batch(dataloader):
    for x_batch, y_batch in dataloader:
        x_batch_np = x_batch.numpy()
        x_batch_processed = np.array([preprocess_eeg(x) for x in x_batch_np])
        x_batch_processed = torch.tensor(x_batch_processed, dtype=torch.float32)
        yield x_batch_processed, y_batch


# モデルの初期化
num_classes = 1854
model = EEGNet(num_classes).to(device)

# 保存したモデルの読み込み
model.load_state_dict(torch.load('MEG_CNN_model_best.pth'))

# 学習の設定
lr = 0.0005
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# 学習率スケジューラーの設定
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# 追加学習の設定
num_epochs = 50
best_val_loss = float('inf')
patience = 10
counter = 0

print('training start')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct_top10 = 0
    total_train = 0

    for x_batch, y_batch in preprocess_batch(dataloader_train):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x_batch.size(0)

        # Calculate top-10 accuracy
        _, top10_pred = outputs.topk(10, dim=1)
        train_correct_top10 += sum([y_batch[i] in top10_pred[i] for i in range(len(y_batch))])

        total_train += y_batch.size(0)

    train_loss /= total_train
    train_accuracy_top10 = train_correct_top10 / total_train

    model.eval()
    val_loss = 0
    val_correct_top10 = 0
    total_val = 0

    with torch.no_grad():
        for x_batch, y_batch in preprocess_batch(dataloader_val):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item() * x_batch.size(0)

            # Calculate top-10 accuracy
            _, top10_pred = outputs.topk(10, dim=1)
            val_correct_top10 += sum([y_batch[i] in top10_pred[i] for i in range(len(y_batch))])

            total_val += y_batch.size(0)

    val_loss /= total_val
    val_accuracy_top10 = val_correct_top10 / total_val

    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Train Loss: {train_loss:.4f}, Train Accuracy (Top-10): {train_accuracy_top10:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Accuracy (Top-10): {val_accuracy_top10:.4f}')

    # Early stopping and model saving
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'MEG_CNN_model_additional_best.pth')
        print("Model saved to MEG_CNN_model_best.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

    # 学習率の調整
    scheduler.step(val_loss)

# モデルの保存
model_path = 'MEG_CNN_model_additional.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')


# 前処理をバッチごとに行う関数
def preprocess_test_batch(dataloader):
    for x_batch in dataloader:
        x_batch_np = x_batch.numpy()
        x_batch_processed = np.array([preprocess_eeg(x) for x in x_batch_np])
        x_batch_processed = torch.tensor(x_batch_processed, dtype=torch.float32)
        yield x_batch_processed


all_predictions = []

with torch.no_grad():
    for x_batch in preprocess_test_batch(dataloader_test):
        x_batch = x_batch.to(device)
        outputs = model(x_batch)
        all_predictions.extend(outputs.cpu().numpy())

# 予測結果の保存
np.save('submission_probabilities.npy', all_predictions)
print('Predictions saved to submission_probabilities.npy')
