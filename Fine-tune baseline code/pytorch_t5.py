# load packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
import time
import datetime
import random
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report,cohen_kappa_score
import re
import matplotlib.pyplot as plt
# import seaborn as sns
# import optuna
# from optuna.pruners import SuccessiveHalvingPruner
# from optuna.samplers import TPESampler


torch.cuda.amp.autocast(enabled=True)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True

# tell pytorch to use cuda
# device = torch.device("cuda")





# prepare and load data
def prepare_df(pkl_location):
    # read pkl as pandas
    df = pd.read_table(pkl_location, header=None,encoding='utf-8',sep='\t', names=['label', 'sentence'])
    # just keep us/kabul labels
    df = df.loc[(df['label'] == 'Exp') | (df['label'] == 'PI') | (df['label'] == 'Sum') | (df['label'] == 'Edu') | (df['label'] == 'QC') | (df['label'] == 'Skill') | (df['label'] == 'Obj')]
    # mask DV to recode
    exp= df['label'] == 'Exp'
    pi= df['label'] == 'PI'
    sum= df['label'] == 'Sum'
    edu= df['label'] == 'Edu'
    qc= df['label'] == 'QC'
    skill= df['label'] == 'Skill'
    obj= df['label'] == 'Obj'
    # reset index
    df = df.reset_index(drop=True)
    return df

# load df
# df = prepare_df('resume_data/数据集/prompt_dataset_csv/test.txt')
# df_train = prepare_df('resume_data/数据集/prompt_dataset_csv/train.txt')
df_train = prepare_df('resume_data/数据集/80000clear.txt')
# df_train=df_train.sample(n=500,random_state=42)

# epochs
epochs = 0

# df_valid = prepare_df('resume_data/数据集/prompt_dataset_csv/valid.txt')
# df_valid=df_valid.sample(n=4,random_state=42)
# df_test = prepare_df('resume_data/数据集/prompt_dataset_csv/test.txt')
# # prepare data
# def clean_df(df):
#     # strip dash but keep a space
#     df['sentence'] = df['sentence'].str.replace('-', ' ')
#     # lower case the data
#     df['sentence'] = df['sentence'].apply(lambda x: x.lower())
#     # remove excess spaces near punctuation
#     df['sentence'] = df['sentence'].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))
#     # generate a word count for sentence
#     df['word_count'] = df['sentence'].apply(lambda x: len(x.split()))
#     # remove excess white spaces
#     df['sentence'] = df['sentence'].apply(lambda x: " ".join(x.split()))
#     # lower case to sentence
#     df['sentence'] = df['sentence'].apply(lambda x: x.lower())
#     # add " </s>" to end of sentence
#     df['sentence'] = df['sentence'] + " </s>"
#     # add " </s>" to end of label
#     df['label'] = df['label'] + " </s>"
#     return df


# # clean df
# df = clean_df(df)


# instantiate T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('pytorch_model/pytorch_t5_large/')

# check token ids
# tokenizer.eos_token_id
# tokenizer.bos_token_id
# tokenizer.unk_token_id
# tokenizer.pad_token_id


# tokenize the main text
def tokenize_corpus(df, tokenizer, max_len):
    # token ID storage
    input_ids = []
    # attension mask storage
    attention_masks = []
    # max len -- 512 is max
    max_len = max_len
    # for every document:
    for doc in df:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            doc,  # document to encode.
                            add_special_tokens=True,  # add tokens relative to model
                            max_length=max_len,  # set max length
                            truncation=True,  # truncate longer messages
                            pad_to_max_length=True,  # add padding
                            return_attention_mask=True,  # create attn. masks
                            return_tensors='pt'  # return pytorch tensors
                       )

        # add the tokenized sentence to the list
        input_ids.append(encoded_dict['input_ids'])

        # and its attention mask (differentiates padding from non-padding)
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


# create tokenized data
sentence_input_ids, sentence_attention_masks = tokenize_corpus(df_train['sentence'].values, tokenizer, 64)
# sentence_input_ids_train, sentence_attention_masks_train = tokenize_corpus(df_train['sentence'].values, tokenizer, 64)
# sentence_input_ids_valid, sentence_attention_masks_valid = tokenize_corpus(df_valid['sentence'].values, tokenizer, 64)
# sentence_input_ids_test, sentence_attention_masks_test = tokenize_corpus(df_test['sentence'].values, tokenizer, 64)

# how long are tokenized labels
ls = []
for i in range(df_train.shape[0]):
    ls.append(len(tokenizer.tokenize(df_train.iloc[i]['label'])))

temp_df = pd.DataFrame({'len_tokens': ls})
temp_df['len_tokens'].mean()  # 2.9

temp_df['len_tokens'].median()  # 3

temp_df['len_tokens'].max()  # 3

label_input_ids, label_attention_masks = tokenize_corpus(df_train['label'].values, tokenizer, 3)
# label_input_ids_train, label_attention_masks_train = tokenize_corpus(df_train['label'].values, tokenizer, 2)
# label_input_ids_valid, label_attention_masks_valid = tokenize_corpus(df_valid['label'].values, tokenizer, 2)
# label_input_ids_test, label_attention_masks_test = tokenize_corpus(df_test['label'].values, tokenizer, 2)


# prepare tensor data sets
def prepare_dataset(sentence_tokens, sentence_masks, label_token, label_masks):
    # create tensor data sets
    tensor_df = TensorDataset(sentence_tokens, sentence_masks, label_token, label_masks)
    # 70% of df
    train_size = int(0.7 * len(df_train))
    # 30% of df
    val_size = len(df_train) - train_size
    # 50% of validation
    test_size = int(val_size - 0.5*val_size)
    # divide the dataset by randomly selecting samples
    train_dataset, val_dataset = random_split(tensor_df, [train_size, val_size],generator=torch.Generator().manual_seed(42))
    # ,generator=torch.Generator().manual_seed(42)
    # divide validation by randomly selecting samples
    val_dataset, test_dataset = random_split(val_dataset, [test_size, test_size+1],generator=torch.Generator().manual_seed(42))

    return train_dataset, val_dataset, test_dataset


# def prepare_dataset(sentence_tokens, sentence_masks, label_token, label_masks):
#     tensor_df = TensorDataset(sentence_tokens, sentence_masks, label_token, label_masks)
#     return tensor_df

# train_dataset=prepare_dataset(sentence_input_ids_train, sentence_attention_masks_train, label_input_ids_train, label_attention_masks_train)
# val_dataset=prepare_dataset(sentence_input_ids_valid, sentence_attention_masks_valid, label_input_ids_valid, label_attention_masks_valid)
# test_dataset=prepare_dataset(sentence_input_ids_test, sentence_attention_masks_test, label_input_ids_test, label_attention_masks_test)

# create tensor data sets
train_dataset, val_dataset, test_dataset = prepare_dataset(sentence_input_ids,
                                                           sentence_attention_masks,
                                                           label_input_ids,
                                                           label_attention_masks
                                                           )


#training models train valid test
def train(model, dataloader, optimizer, device=None):

    # capture time
    total_t0 = time.time()

    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    print('Training...')

    # reset total loss for epoch
    train_total_loss = 0
    total_train_f1 = 0

    # put model into traning mode
    model.train()
    CUDA_LAUNCH_BLOCKING=1
    # for each batch of training data...
    for step, batch in enumerate(dataloader):

        # progress update every 40 batches.
        if step % 40 == 0 and not step == 0:

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))

        # Unpack this training batch from our dataloader:
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input tokens
        #   [1]: attention masks
        #   [2]: label tokens
        #   [3]: label attenion masks
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_label_ids = batch[2].to(device)
        b_label_mask = batch[3].to(device)

        # clear previously calculated gradients
        optimizer.zero_grad()

        # runs the forward pass with autocasting.
        # with autocast():
            # forward propagation (evaluate model on training batch)
        outputs = model(input_ids=b_input_ids,
                        attention_mask=b_input_mask,
                        labels=b_label_ids,
                        decoder_attention_mask=b_label_mask)

        loss,prediction_scores = outputs[:2]
        # loss = loss.repeat(2)
        # sum the training loss over all batches for average loss at end
        # loss is a tensor containing a single value

        loss = loss.unsqueeze(0)
        loss = loss.mean()

        train_total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
  
        # # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # # Backward passes under autocast are not recommended.
        # # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        # scaler.scale(loss).backward()

        # # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # # otherwise, optimizer.step() is skipped.
        # scaler.step(optimizer)

        # # Updates the scale for next iteration.
        # scaler.update()

        # # update the learning rate
        # scheduler.step()

    # calculate the average loss over all of the batches
    avg_train_loss = train_total_loss / len(dataloader)

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'Train Loss': avg_train_loss
        }
    )

    # training time end
    training_time = format_time(time.time() - total_t0)

    # print result summaries
    print("")
    print("summary results")
    print("epoch | trn loss | trn time ")
    print(f"{epoch+1:5d} | {avg_train_loss:.5f} | {training_time:}")

    return training_stats


def validating(model, dataloader, device=None):

    # capture validation time
    total_t0 = time.time()

    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("")
    print("Running Validation...")

    # put the model in evaluation mode
    model.eval()

    # track variables
    total_valid_loss = 0

    # evaluate data for one epoch
    for batch in dataloader:

        # Unpack this training batch from our dataloader:
        # `batch` contains three pytorch tensors:
        #   [0]: input tokens
        #   [1]: attention masks
        #   [2]: label tokens
        #   [3]: label attenion masks
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_label_ids = batch[2].to(device)
        b_label_mask = batch[3].to(device)
        CUDA_LAUNCH_BLOCKING=1

        # tell pytorch not to bother calculating gradients
        # as its only necessary for training
        with torch.no_grad():

            # forward propagation (evaluate model on training batch)
            outputs = model(input_ids=b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_label_ids,
                            decoder_attention_mask=b_label_mask)

            loss, prediction_scores = outputs[:2]

            # sum the training loss over all batches for average loss at end
            # loss is a tensor containing a single value
            
            loss = loss.unsqueeze(0)
            loss = loss.mean()

            total_valid_loss += loss.item()

    # calculate the average loss over all of the batches.
    global avg_val_loss
    avg_val_loss = total_valid_loss / len(dataloader)

    # Record all statistics from this epoch.
    valid_stats.append(
        {
            'Val Loss': avg_val_loss,
            'Val PPL.': np.exp(avg_val_loss)
        }
    )

    # capture end validation time
    training_time = format_time(time.time() - total_t0)

    # print result summaries
    print("")
    print("summary results")
    print("epoch | val loss | val ppl | val time")
    print(f"{epoch+1:5d} | {avg_val_loss:.5f} | {np.exp(avg_val_loss):.3f} | {training_time:}")

    return valid_stats


# 绘制混淆矩阵
conf_matrix = torch.zeros((7, 7))

from matplotlib import pyplot as plt
import numpy as np
#confuse matrix
import itertools
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        try:
            conf_matrix[p, t] += 1
        except Exception as e:
            pass
        continue
                    
    return conf_matrix

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




    

# time function
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))






# instantiate model T5 transformer with a language modeling head on top
model = T5ForConditionalGeneration.from_pretrained('pytorch_model/pytorch_t5_large/')  # to GPU
# C:/Users/gan chengguang/Desktop/pytorch_T5_3B   pytorch_model/pytorch_t5_large/
device = torch.device('cuda:0')
device_ids=[0,1]
model = torch.nn.DataParallel(model, device_ids=device_ids)

model = model.cuda(device=device_ids[0])



# create DataLoaders with samplers
train_dataloader = DataLoader(train_dataset,
                              batch_size=64,
                              shuffle=False)

valid_dataloader = DataLoader(val_dataset,
                              batch_size=64,
                              shuffle=False)

test_dataloader = DataLoader(test_dataset,
                              batch_size=128,
                              shuffle=False)


# Adam w/ Weight Decay Fix
# set to optimizer_grouped_parameters or model.parameters()
optimizer = torch.optim.AdamW(params=model.parameters(),
                  lr = 1e-5
                  )



# # lr scheduler
# total_steps = len(train_dataloader) * epochs
# scheduler = get_linear_schedule_with_warmup(optimizer,
#                                             num_warmup_steps=0,
#                                             num_training_steps=total_steps)

# # create gradient scaler for mixed precision
# scaler = GradScaler()

# caculate paramate 
# 定义总参数量、可训练参数量及非可训练参数量变量
Total_params = 0
Trainable_params = 0
NonTrainable_params = 0

# 遍历model.parameters()返回的全局参数列表
for param in model.parameters():
    mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
    Total_params += mulValue  # 总参数量
    if param.requires_grad:
        Trainable_params += mulValue  # 可训练参数量
    else:
        NonTrainable_params += mulValue  # 非可训练参数量

print(f'Total params: {Total_params}')
print(f'Trainable params: {Trainable_params}')
print(f'Non-trainable params: {NonTrainable_params}')




# create training result storage
training_stats = []
valid_stats = []
best_valid_loss = float('inf')




# for each epoch
for epoch in range(epochs):
    # train
    train(model, train_dataloader, optimizer)
    # validate
    validating(model, valid_dataloader)
    # check validation loss
    # if valid_stats[epoch]['Val Loss'] < best_valid_loss:
    #     best_valid_loss = valid_stats[epoch]['Val Loss']
    #     # save best model for use later
    #     torch.save(model.state_dict(), 't5-classification.pt')  # torch save
    #     model_to_save = model.module if hasattr(model, 'module') else model
    #     model_to_save.save_pretrained('./model_save/t5-classification/')  # transformers save
    #     tokenizer.save_pretrained('./model_save/t5-classification/')  # transformers save


# pd.set_option('precision', 3)
df_train_stats = pd.DataFrame(data=training_stats)
df_valid_stats = pd.DataFrame(data=valid_stats)
df_stats = pd.concat([df_train_stats, df_valid_stats], axis=1)
df_stats.insert(0, 'Epoch', range(1, len(df_stats)+1))
df_stats = df_stats.set_index('Epoch')
df_stats




device = torch.device("cuda")


def testing(model, dataloader):

    print("")
    print("Running Testing...")

    # measure training time
    t0 = time.time()

    # put the model in evaluation mode
    model.eval()

    # track variables
    total_test_loss = 0
    total_test_acc = 0
    total_test_f1 = 0
    predictions = []
    actuals = []
    total = 0
    correct = 0
    CUDA_LAUNCH_BLOCKING=1

    # evaluate data for one epoch
    for step, batch in enumerate(dataloader):
        # progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

        # Unpack this training batch from our dataloader:
        # `batch` contains three pytorch tensors:
        #   [0]: input tokens
        #   [1]: attention masks
        #   [2]: label tokens
        #   [3]: label attenion masks
        b_input_ids = batch[0].to('cuda')
        b_input_mask = batch[1].to('cuda')
        b_label_ids = batch[2].to('cuda')
        b_label_mask = batch[3].to('cuda')

        # tell pytorch not to bother calculating gradients
        # as its only necessary for training
        with torch.no_grad():

            # forward propagation (evaluate model on training batch)
            outputs = model(input_ids=b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_label_ids,
                            decoder_attention_mask=b_label_mask)

            loss, prediction_scores = outputs[:2]
            
            loss = loss.unsqueeze(0)
            loss = loss.mean()

            total_test_loss += loss.item()

            generated_ids = model.generate(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask,
                    max_length=3
                    )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            label = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in b_label_ids]
            preds[:0]        
            total_test_acc += accuracy_score(label, preds)
            total_test_f1 += f1_score(preds, label,
                                       average='weighted',
                                       labels=np.unique(preds))

            predictions.extend(preds)
            actuals.extend(label)
            # print(label)
            # print(len(predictions))
            line_label = {'Exp':0 ,'PI':1,'Sum':2,'Edu':3,'QC':4,'Skill':5,'Obj':6}
            total = pd.Series(label).map(line_label).tolist()
            line_preds = {'Exp':0 ,'PI':1,'Sum':2,'Edu':3,'QC':4,'Skill':5,'Obj':6}
            correct = pd.Series(preds).map(line_preds).tolist()
            # print(correct,total)
            global conf_matrix
            conf_matrix = confusion_matrix(preds=correct, labels=total, conf_matrix=conf_matrix)
            # print(conf_matrix.shape)


    # calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(dataloader)

    avg_test_acc = total_test_acc / len(test_dataloader)

    avg_test_f1 = total_test_f1 / len(test_dataloader)

    # Record all statistics from this epoch.
    test_stats.append(
        {
            'Test Loss': avg_test_loss,
            'Test PPL.': np.exp(avg_test_loss),
            'Test Acc.': avg_test_acc,
            'Test F1': avg_test_f1
        }
    )
    global df2
    temp_data = pd.DataFrame({'predicted': predictions, 'actual': actuals})
    df2 = df2.append(temp_data)
    kappa_score=cohen_kappa_score(actuals,predictions)
    print("kappa score %f" % kappa_score)
    

    # print(classification_report(correct, total,digits=4))
    return test_stats


df2 = pd.DataFrame({'predicted': [], 'actual': []})
test_stats = []
# model.load_state_dict(torch.load('C:/Users/Administrator/iCloudDrive/Desktop/pytorch/t5-classification.pt'))
testing(model, test_dataloader) 
# 0-shot [{'Test Loss': 14.486225780206746, 'Test PPL.': 1955635.5345919845, 'Test Acc.': 0.15027289669861557, 'Test F1': 0.08045841004025243}]
df_test_stats = pd.DataFrame(data=test_stats)
print(df_test_stats)


attack_types = ['Exp','PI','Sum','Edu','QC','Skill','Obj']
plot_confusion_matrix(conf_matrix.numpy(), classes=attack_types, normalize=False,
                                 title='Normalized confusion matrix')