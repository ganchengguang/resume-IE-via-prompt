#! -*- coding:utf-8 -*-
# 句子对分类任务，LCQMC数据集
# val_acc: 0.887071, test_acc: 0.870320
import numpy as np
import os

from tensorflow.core.framework.summary_pb2 import Summary
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
batch_size = 8
config_path = 'E:/bert4keras-master/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'E:/bert4keras-master/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'E:/bert4keras-master/uncased_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)



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

model.load_weights('E:/bert4keras-master/best_model.weights')



#predict_sentence('· C.A.I.I.B. conducted by Indian Institute of Bankers')
#collected and text preprocess
import docx
import os
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


def pdf_to_text(fname, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = open(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close()
    result = []
    for line in text.split('\\n'):
        line2 = line.strip()
        if line2 != '':
            result.append(line2)
    return result

def preprocess_text(text):
    text = ' '.join(text.split())
    text = join_name_tag(text)
    return text

def join_name_tag(text):
    text = text.replace('\\u2003', '')
    return text
data_dir_path = 'C:\\Users\\Administrator\\iCloudDrive\\Desktop\\论文模型\\keras-english-resume-parser-and-analyzer-master\\demo\\data\\resume_samples\\(NEW)2043_RAKESH_KUMAR_GUPTA (1).docx'
def predict_sentence(sentence):
            token_ids, segment_ids = tokenizer.encode(
                        sentence, maxlen=len(sentence.split())
                    )
            y_pred = model.predict([[token_ids],[segment_ids]]).argmax(axis=1)
            print(line_label[y_pred[0]],':',sentence)
            return line_label[y_pred[0]]

def docx_to_text(file_path):
    doc = docx.Document(file_path)
    result = []
    for p in doc.paragraphs:
        txt = p.text.strip()
        if txt != '':
            txt = preprocess_text(txt)
            predict_sentence(txt)
            result.append(txt)
    return result

#main 
docx_to_text(data_dir_path)




#test
#predict_sentence('· C.A.I.I.B. conducted by Indian Institute of Bankers')