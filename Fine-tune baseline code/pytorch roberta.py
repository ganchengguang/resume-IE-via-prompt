#coding: utf-8
from logging import critical
import pandas as pd
from sklearn.model_selection import train_test_split

# データの読込
df = pd.read_table('80000clear.txt', encoding='utf-8',sep='\t', names=['label', 'sentence'])
# df = pd.read_table('/train.txt', encoding='utf-8',sep='\t', names=['label', 'sentence'])
# df=df.sample(n=10000,random_state=42)
# train=df
# print(df)

max_len = 64
DROP_RATE = 0.1
OUTPUT_SIZE = 7
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-5


# データの抽出
#df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'label']]

# データの分割
train, valid_test = train_test_split(df, test_size=0.3, shuffle=True, random_state=42, stratify=df['label'])
#print(valid_test.head), stratify=valid_test['label'], stratify=df['label']
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=42, stratify=valid_test['label'])
# valid=pd.read_table('valid.txt', encoding='utf-8',sep='\t', names=['label', 'sentence'])
# valid=valid.sample(n=512,random_state=42)
# test=pd.read_table('test.txt', encoding='utf-8',sep='\t', names=['label', 'sentence'])

train.reset_index(drop=True, inplace=True)
valid.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

#print(train.head())
import numpy as np
# 事例数の確認
#print('【学習データ】')
#print(train['label'].value_counts())
list_shape = np.array(train).shape
#print('【検証データ】')
#print(valid['label'].value_counts())
list_shapev = np.array(valid).shape
print('【評価データ】')
# #print(test['label'].value_counts())
list_shapet = np.array(test).shape


print('trainsize',list_shape)

print('validsize',list_shapev)

print('testsize',list_shapet)



import numpy as np
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from torch import optim
from torch import cuda
import time
from matplotlib import pyplot as plt

# Datasetの定義
class CreateDataset(Dataset):
  def __init__(self, X, y, tokenizer, max_len):
    self.X = X
    self.y = y
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):  # len(Dataset)で返す値を指定
    return len(self.y)

  def __getitem__(self, index):  # Dataset[index]で返す値を指定
    text = self.X[index]
    inputs = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      pad_to_max_length=True,
    )
    ids = inputs['input_ids']
    mask = inputs['attention_mask']

    return {
      'ids': torch.LongTensor(ids),
      'mask': torch.LongTensor(mask),
      'labels': torch.Tensor(self.y[index]),
      'text': text
      # 'labels': torch.tensor(self.y[index],dtype=torch.int64)
    }

    # 正解ラベルのone-hot化
y_train = pd.get_dummies(train, columns=['label'])[['label_Exp','label_PI','label_Sum','label_Edu', 'label_QC','label_Skill', 'label_Obj']].values
# 位置4的QC插入一列0
# y_train = np.hstack([y_train[:,:4], np.zeros((y_train.shape[0],1)), y_train[:,4:]])
y_valid = pd.get_dummies(valid, columns=['label'])[['label_Exp','label_PI','label_Sum','label_Edu', 'label_QC', 'label_Skill', 'label_Obj']].values
y_test = pd.get_dummies(test, columns=['label'])[['label_Exp','label_PI','label_Sum','label_Edu', 'label_QC', 'label_Skill', 'label_Obj']].values

# y_trains = [np.argmax(item) for item in y_train]
# y_valids = [np.argmax(item) for item in y_valid]
# y_tests = [np.argmax(item) for item in y_test]
# n=1
# y_train = [y_trains[i:i+n] for i in range(0, len(y_trains),n)]
# y_valid = [y_valids[i:i+n] for i in range(0, len(y_valids),n)]
# y_test = [y_tests[i:i+n] for i in range(0, len(y_valids),n)]
# Datasetの作成




tokenizer = RobertaTokenizer.from_pretrained('pytorch_model/pytorch_roberta_large/')
dataset_train = CreateDataset(train['sentence'], y_train, tokenizer, max_len)
dataset_valid = CreateDataset(valid['sentence'], y_valid, tokenizer, max_len)
dataset_test = CreateDataset(test['sentence'], y_test, tokenizer, max_len)


for var in dataset_train[0]:
  print(f'{var}: {dataset_train[0][var]}')


  # BERT分類モデルの定義
class RobertaClass(torch.nn.Module):
  def __init__(self, drop_rate, output_size):
    super().__init__()
    self.bert = RobertaModel.from_pretrained('pytorch_model/pytorch_roberta_large/')
    self.drop = torch.nn.Dropout(drop_rate)
    self.fc = torch.nn.Linear(1024, output_size)  # BERTの出力に合わせて768次元を指定

  def forward(self, ids, mask):
    _, out = self.bert(ids, attention_mask=mask, return_dict=False)
    out = self.fc(self.drop(out))
    return out


    # モデルの学習
def calculate_loss_and_accuracy(model, criterion, loader, device):
  ### 損失・正解率を計算###
  model.eval()
  loss = 0.0
  total = 0
  correct = 0
  with torch.no_grad():
    for data in loader:
      # デバイスの指定
      ids = data['ids'].to(device)
      mask = data['mask'].to(device)
      labels = data['labels'].to(device)

      # 順伝播
      outputs = model(ids, mask)

      # 損失計算
      loss += criterion(outputs, labels).item()

      # 正解率計算
      pred = torch.argmax(outputs, dim=-1).cpu().numpy() # バッチサイズの長さの予測ラベル配列
      labels = torch.argmax(labels, dim=-1).cpu().numpy()  # バッチサイズの長さの正解ラベル配列
      total += len(labels)
      correct += (pred == labels).sum().item()

  return loss / len(loader), correct / total


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, device=None):
  # デバイスの指定
  model.to(device)

  # dataloaderの作成
  dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
  dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

  # 学習
  log_train = []
  log_valid = []
  for epoch in range(num_epochs):
    # 開始時刻の記録
    s_time = time.time()

    # 訓練モードに設定
    model.train()
    for data in dataloader_train:
      # デバイスの指定
      ids = data['ids'].to(device)
      mask = data['mask'].to(device)
      labels = data['labels'].to(device)

      # 勾配をゼロで初期化
      optimizer.zero_grad()

      # 順伝播 + 誤差逆伝播 + 重み更新
      outputs = model(ids, mask)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      # e_time = time.time()
      # log_train.append(loss.item())
      # print("\r%f" % loss, end='')
      # print(f'epoch: {epoch + 1},{(e_time - s_time):.4f}sec')

    loss_train, acc_train = calculate_loss_and_accuracy(model, criterion, dataloader_train, device)
    loss_valid, acc_valid = calculate_loss_and_accuracy(model, criterion, dataloader_valid, device)
    log_train.append([loss_train, acc_train])
    log_valid.append([loss_valid, acc_valid])
    #torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')
    e_time = time.time()
    print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(e_time - s_time):.4f}sec') 
  return {'train': log_train, 'valid': log_valid}
      
      
    # パラメータの設定


# モデルの定義
model = RobertaClass(DROP_RATE, OUTPUT_SIZE)

#from torchvision import models
#model = models.model()
#print(model)

#model.num_parameters()

# 損失関数の定義
pos_weight = torch.tensor([1.916281559,
5.882186053,
12.02105584,
8.271496063,
80.88911704,
15.79510826,
35.21949039
])
device = torch.device('cuda:0')
pos_weight = pos_weight.to(device)
from pytorch_loss import focal_loss , Focal_Loss
from unbalanced_loss.dice_loss_nlp import MultiDSCLoss
from unbalanced_loss.focal_loss import MultiFocalLoss
# criterion = MultiFocalLoss(num_class=7,gamma=2.0, reduction='mean')
# criterion = MultiDSCLoss(alpha= 0., smooth= 1.0, reduction = "mean")
criterion = torch.nn.BCEWithLogitsLoss()
# criterion = torch.nn.CrossEntropyLoss(weight=pos_weight)

# オプティマイザの定義
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

# デバイスの指定
device = 'cuda' #if cuda.is_available() else 'cpu'

# モデルの学習
log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, device=device)

# ログの可視化
# x_axis = [x for x in range(1, len(log['train']) + 1)]
# fig, ax = plt.subplots(1, 2, figsize=(15, 5))
# ax[0].plot(x_axis, np.array(log['train']).T[0], label='train')
# ax[0].plot(x_axis, np.array(log['valid']).T[0], label='valid')
# ax[0].set_xlabel('epoch')
# ax[0].set_ylabel('loss')
# ax[0].legend()
# ax[1].plot(x_axis, np.array(log['train']).T[1], label='train')
# ax[1].plot(x_axis, np.array(log['valid']).T[1], label='valid')
# ax[1].set_xlabel('epoch')
# ax[1].set_ylabel('accuracy')
# ax[1].legend()
# plt.show()

# 正解率の算出
def calculate_accuracy(model, dataset, device):
  # Dataloaderの作成
  batch_size=64
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

  model.eval()
  total = 0
  correct = 0
  with torch.no_grad():
    for data in loader:
      # デバイスの指定
      ids = data['ids'].to(device)
      mask = data['mask'].to(device)
      labels = data['labels'].to(device)

      # 順伝播 + 予測値の取得 + 正解数のカウント
      outputs = model.forward(ids, mask)
      pred = torch.argmax(outputs, dim=-1).cpu().numpy()
      labels = torch.argmax(labels, dim=-1).cpu().numpy()
      total += len(labels)
      correct += (pred == labels).sum().item()

  return correct / total

print(f'正解率（学習データ）：{calculate_accuracy(model, dataset_train, device):.3f}')
print(f'正解率（検証データ）：{calculate_accuracy(model, dataset_valid, device):.3f}')
print(f'正解率（評価データ）：{calculate_accuracy(model, dataset_test, device):.3f}')




# confuse matrix
import itertools
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

    # 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
    # x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(i, j, num,
                verticalalignment='center',
                horizontalalignment="center",
                color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

conf_matrix = torch.zeros((7, 7))
def calculate_accuracy(model, dataset, device):
  # Dataloaderの作成
  batch_size=64
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

  model.eval()
  total = 0
  correct = 0
  with torch.no_grad():
    for data in loader:
      # デバイスの指定
      ids = data['ids'].to(device)
      mask = data['mask'].to(device)
      labels = data['labels'].to(device)

      # 順伝播 + 予測値の取得 + 正解数のカウント
      outputs = model.forward(ids, mask)
      preds = torch.argmax(outputs, dim=-1).cpu().numpy()
      labels = torch.argmax(labels, dim=-1).cpu().numpy()
      # total += len(labels)
      # correct += (pred == labels).sum().item()
      global conf_matrix
      conf_matrix = confusion_matrix(preds, labels=labels, conf_matrix=conf_matrix) 
  # return pred
calculate_accuracy(model, dataset_test, device)

# conf_matrix需要是numpy格式
# attack_types是分类实验的类别，eg：
attack_types = ['Exp','PI','Sum','Edu','QC','Skill','Obj']
plot_confusion_matrix(conf_matrix.numpy(), classes=attack_types, normalize=False,
                                 title='Normalized confusion matrix')





label_dict = {0: 'Exp', 1: 'PI', 2: 'Sum', 3: 'Edu', 4: 'QC', 5: 'Skill', 6: 'Obj'}


# f1 recall 計算
from sklearn.metrics import accuracy_score, recall_score, precision_score,f1_score,classification_report
def calculate_accuracy(model, dataset, output_file_path, device):
  # Dataloaderの作成
  batch_size=64
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

  model.eval()
  total = np.array([])
  correct = np.array([])
  text = []
  with torch.no_grad():
    for data in loader:
      # デバイスの指定
      ids = data['ids'].to(device)
      mask = data['mask'].to(device)
      labels = data['labels'].to(device)

      # 順伝播 + 予測値の取得 + 正解数のカウント
      outputs = model.forward(ids, mask)
      preds = torch.argmax(outputs, dim=-1).cpu().numpy()
      labels = torch.argmax(labels, dim=-1).cpu().numpy()
      # total += len(labels)
      # correct += (preds == labels).sum().item()
      correct = np.concatenate((correct, preds))
      total = np.concatenate((total, labels))
      text += data['text']
    
    total = [label_dict[int(x)] for x in total]
    correct = [label_dict[int(x)] for x in correct]


    with open(output_file_path, 'w', encoding='utf-8') as f:
          for t, p, s in zip(total, correct, text):
              f.write(f'{t}\t{p}\t{s}\n')
    print(total) 
    print(correct)
    acc = precision_score(total, correct,average='weighted')
    recall = recall_score(total, correct, average='micro')
    f1score = f1_score(total, correct, average='weighted')
    print('acc',acc)
    print('recall',recall)
    print('f1score',f1score)
    
    print(classification_report(total, correct,digits=4))
    
    
    return recall
output_file_path = 'predictions.txt'
calculate_accuracy(model, dataset_test, output_file_path, device)
