
from openprompt.data_utils.text_classification_dataset import AgnewsProcessor,resumeProcessor

dataset={}
dataset['train'] = resumeProcessor().get_train_examples("prompt_dataset_csv")
# We sample a few examples to form the few-shot training pool
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.utils.reproduciblity import set_seed

sampler  = FewShotSampler(num_examples_total=50, also_sample_dev=False)


import random
import numpy as np
import torch
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

dataset['train'] = sampler(dataset['train'], seed=42)
# dataset['validation'] = resumeProcessor().get_test_examples("prompt_dataset_csv")
dataset['test'] = resumeProcessor().get_test_examples("prompt_dataset_csv")

dataset['train']

from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "pytorch_model/pytorch_roberta_large")


from openprompt.prompts.manual_template import resumeManualTemplate
from openprompt.prompts import SoftTemplate
# mytemplate = resumeManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_a"} In this sentence, the topic is {"mask"}.')
# 0.3473
# mytemplate = resumeManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_a"} this sentence is talking about {"mask"}.')
# 0.2828
# mytemplate = resumeManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_a"} this sentence belongs in the {"mask"} section of the resumes.')
# resume+s 0.4160   resume no s 0.3966  curriculum vitae 0.2719


epoch=4



# resume+s 0.61   resume no s 0.6132  curriculum vitae 0.6209 belong+s 0.6209  belong no s 0.584
mytemplate = resumeManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_a"} this sentence belongs in the {"mask"} section of the curriculum vitae.')
mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, num_tokens=20).from_file(f"C:/Users/Administrator/iCloudDrive/Desktop/pytorch/resume_data/script/soft template.txt")
# mytemplate = PtuningTemplate(model=plm,tokenizer=tokenizer,prompt_encoder_type="lstm").from_file(f"C:/Users/Administrator/iCloudDrive/Desktop/pytorch/resume_data/script/soft template.txt")
# mytemplate = PrefixTuningTemplate(model=plm,tokenizer=tokenizer).from_file(f"C:/Users/Administrator/iCloudDrive/Desktop/pytorch/resume_data/script/soft template.txt")


wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)

from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=64, 
    batch_size=4,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
# next(iter(train_dataloader))

# ## Define the verbalizer
# In classification, you need to define your verbalizer, which is a mapping from logits on the vocabulary to the final label probability. Let's have a look at the verbalizer details:

from openprompt.prompts import SoftVerbalizer,KnowledgeableVerbalizer,PTRVerbalizer
import torch


myverbalizer = KnowledgeableVerbalizer(tokenizer, num_classes=7).from_file("resume_data/script/KnowledgeableVerbalizeroriginal.txt")
# myverbalizer=ManualVerbalizer(tokenizer,num_classes=7,label_words=[["experience"],  ["personal information"], ["summary"], ["education"], ["qualification"], ["skill"], ["object"]])
myverbalizer = SoftVerbalizer(tokenizer, plm, num_classes=7)
# myverbalizer=AutomaticVerbalizer(tokenizer,num_candidates=10,num_classes=7,score_fct='llr',balance=True)


from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()

# ## below is standard training


from transformers import  AdamW, get_linear_schedule_with_warmup
loss_func = torch.nn.CrossEntropyLoss()

no_decay = ['bias', 'LayerNorm.weight']

# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# Using different optimizer for prompt parameters and model parameters

optimizer_grouped_parameters2 = [
    {'params': prompt_model.verbalizer, "lr":3e-5},
    # {'params': prompt_model.verbalizer.group_parameters_2, "lr":3e-4},
]


optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
# optimizer2 = AdamW(optimizer_grouped_parameters2)


for epoch in range(epoch):
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer1.step()
        optimizer1.zero_grad()
        # optimizer2.step()
        # optimizer2.zero_grad()
        print(tot_loss/(step+1))

# ## evaluate

# %%
# validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
#     tokenizer_wrapper_class=WrapperClass, max_seq_length=64,
#     batch_size=2,shuffle=False, teacher_forcing=False, predict_eos_token=False,
#     truncate_method="head")

# prompt_model.eval()

# allpreds = []
# alllabels = []
# for step, inputs in enumerate(validation_dataloader):
#     if use_cuda:
#         inputs = inputs.cuda()
#     logits = prompt_model(inputs)
#     labels = inputs['label']
#     alllabels.extend(labels.cpu().tolist())
#     allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

# acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
# print("validation:",acc)



from matplotlib import pyplot as plt
import numpy as np
#confuse matrix
import itertools
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


# construct confuse matrix
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

    plt.axis("equal")
    ax = plt.gca()  
    left, right = plt.xlim() 
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

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


from sklearn.metrics import accuracy_score, recall_score, precision_score,f1_score,classification_report

test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=64,
    batch_size=32,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
allpreds = []
alllabels = []
for step, inputs in enumerate(test_dataloader):
    if use_cuda:
        inputs = inputs.cuda()
    logits = prompt_model(inputs)
    labels = inputs['label']
    preds = torch.argmax(logits, dim=-1).cpu().tolist()
    # print(preds,labels)
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    conf_matrix = confusion_matrix(preds, labels=labels, conf_matrix=conf_matrix) 
f1score = f1_score(alllabels, allpreds, average='micro')
print("f1:",f1score)
acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
print("test:", acc)  


# conf_matrix需要是numpy格式
# attack_types是分类实验的类别，eg：
attack_types = ['Exp','PI','Sum','Edu','QC','Skill','Obj']
plot_confusion_matrix(conf_matrix.numpy(), classes=attack_types, normalize=False,
                                 title='Normalized confusion matrix')

