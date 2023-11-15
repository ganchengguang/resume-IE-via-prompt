# resume-IE-via-prompt
paper code
This code is following paper has been used.
A Few-shot Approach to Resume Information Extraction via Prompts
Chengguang Gan, Tatsunori Mori
https://arxiv.org/abs/2209.09450

And you can found seven classes resume dataset in following website.
https://www.kaggle.com/datasets/chingkuangkam/resume-seven-class or 
https://huggingface.co/datasets/ganchengguang/resume_seven_class

NOTICE! If you want use code with resume dataset. You must be change a series set of openprompt framework. Make openprompt framework can correct read the dataset format line by line. The orginal openprompt just can read AG's News. You need change that adapt to resume dataset's format.
!!!     You can found the replace code in OpenPromtCustomSourceCode.py file. Please Following the instruction to replace all of OpenPrompt SouceCode. Then you can adapt the seven-class resume dataset.
The script folder include Knowledgeable verbalizer's script/
