import numpy as np
import os
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense
from tqdm import tqdm

set_gelu('tanh')  # 切换gelu版本
line_label = {0: 'experience', 1: 'knowledge', 2: 'education', 3: 'project', 4: 'others'}
label2index = {v:k for k,v in line_label.items()}

print(label2index)

maxlen = 128
batch_size = 16
config_path = 'E:/bert4keras-master/wwm_uncased_L-24_H-1024_A-16/bert_config.json'
checkpoint_path = 'E:/bert4keras-master/wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt'
dict_path = 'E:/bert4keras-master/wwm_uncased_L-24_H-1024_A-16/vocab.txt'




def load_text_label_pairs(data_dir_path, label_type=None):
    if label_type is None:
        label_type = 'line_type'

    result = []

    for f in os.listdir(data_dir_path):
        data_file_path = os.path.join(data_dir_path, f)
        if os.path.isfile(data_file_path) and f.lower().endswith('.txt'):
            with open(data_file_path, mode='rt', encoding='utf8') as file:
                for line in file:
                    
                    line_type, line_label, sentence = line.strip().split('\t')
                    if line_label not in label2index.keys():
                        continue
                    if label_type == 'line_type':
                        result.append((sentence, line_type))
                    else:
                        result.append((sentence, label2index[line_label]))
    return result
# 加载数据集


train_data = load_text_label_pairs('E:/bert4keras-master/resume_data_73/train', label_type='line_label')
valid_data = load_text_label_pairs('E:/bert4keras-master/resume_data_73/valid', label_type='line_label')
test_data = load_text_label_pairs('E:/bert4keras-master/resume_data_73/test', label_type='line_label')

list_shape = np.array(train_data).shape
print('trainsize',list_shape)

list_shapev = np.array(valid_data).shape
print('validsize',list_shapev)

list_shapet = np.array(test_data).shape
print('testsize',list_shapet)
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (sentence, line_label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                sentence, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([line_label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    return_keras_model=False,
)

output = Dropout(rate=0.1)(bert.model.output)
output = Dense(
    units=5, activation='softmax', kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

print(train_generator)


#confuse matrix
# 混淆矩阵定义
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys,#这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
                          normalize=True):
   
 
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
	#plt.savefig('/content/drive/My Drive/Colab Notebooks/confusionmatrix32.png',dpi=350)
    plt.show()
# 显示混淆矩阵
def plot_confuse(model,data):
    all_preds = np.array([])
    all_true = np.array([])
    for x_true, y_true in data:

        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        
        all_preds = np.concatenate((all_preds, y_pred))
        all_true = np.concatenate((all_true, y_true))
    conf_mat = confusion_matrix(all_true, all_preds)
    plt.figure()
    plot_confusion_matrix(conf_mat, normalize=False,target_names=labels,title='Confusion Matrix')
#=========================================================================================
#最后调用这个函数即可。 test_x是测试数据，test_y是测试标签（这里用的是One——hot向量）
#labels是一个列表，存储了你的各个类别的名字，最后会显示在横纵轴上。
#比如这里我的labels列表
labels=['experience','knowledge','education','project','others']


import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score,f1_score,classification_report
def evaluate(data):
    pre,recall, right = 0., 0., 0.
    all_preds = np.array([])
    all_true = np.array([])
    for x_true, y_true in data:

        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        
        all_preds = np.concatenate((all_preds, y_pred))
        all_true = np.concatenate((all_true, y_true))
        #pre += len(y_true)
        #recall += (len(y_true)+len(x_true))
        #right += (y_true == y_pred).sum()
    
    acc = precision_score(all_true, all_preds,average='weighted')
    recall = recall_score(all_true, all_preds, average='micro')
    f1score = f1_score(all_true, all_preds, average='weighted')
    print('acc',acc)
    print('recall',recall)
    print('f1score',f1score)
    
    print(classification_report(all_true, all_preds,digits=4))
    
    
    return recall
    #return (2*(right/pre)*(right/recall))/(right/pre+right/recall)


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )
      


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=20,
        callbacks=[evaluator]
    )

    model.load_weights('best_model.weights')
    print(u'final test acc: %05f\n' % (evaluate(test_generator)))
    plot_confuse(model, test_generator)
else:

    model.load_weights('best_model.weights')