import os
import sys
sys.path.append('..')
from utils import *
from dataset import LoadDataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import glob
os.environ["TOKENIZERS_PARALLELISM"] = "false"
 # os.environ ['CUDA_VISIBLE_DEVICES'] = '0'

from tqdm import tqdm
import time


if __name__ == '__main__':
    setup_seed(0)
    tokenizer, config = get_TokenizerConfig()
    Apmodel = load_model(config, use_property=True, CUDA=True)
    Apmodel.load_state_dict(torch.load('../trained_model/Bert/Ap/Ap_cls_fewfeature_6060train_1fold_0303.pth'))  # 使用哪个模型，可以修改
    cppmodel = load_model(config, use_property=True, CUDA=True)
    cppmodel.load_state_dict(torch.load('../trained_model/Bert/cpp/cpp_cls_fewfeature_3936train_3fold_0303.pth'))  # 使用哪个模型，可以修改

    # 待筛选文件所在文件夹，需要修改，末尾有无\都可以, 此时的文件应该是包括氨基酸序列，多种基本的理化性质
    library_path = r'E:\LS\深度学习\code\code_new_0815\practice-cxy\理化性质\*.csv'  # *号后面加文件类型

    # 运行完后，结果保存的文件夹，这个文件夹可以不用手动创建，如果没有这个文件夹，代码会生成, 需要修改，末尾有无\都可以
    save_path = r'E:\LS\深度学习\code\code_new_0815\practice-cxy\transformer'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for csv_name in glob.glob(library_path):
        file = os.path.join(library_path, csv_name)
        print(file)
        data = pd.read_csv(file,index_col=0)
        mydataset = LoadDataset.ScreenDataset(data, tokenizer, use_property=True)
        mydataloader = DataLoader(
                        mydataset,
                        batch_size=128, # 这个参数是指一次对多少条序列进行评分，数值需要根据电脑而定
                        num_workers=0, # 这个参数简单来说指采用多少线程来加载处理数据，数值需要根据电脑而定
                        pin_memory=True,
                        shuffle=False
                    )
        Ap_all_predict = []
        cpp_all_predict = []
        with torch.no_grad():
            Apmodel.eval()
            cppmodel.eval()
            for i, inputs in enumerate(tqdm(mydataloader)):
                Apoutput = Apmodel(input_ids=inputs['input_ids'].cuda(),
                               feature=inputs['feature'].cuda(),
                               attention_mask=inputs['attention_mask'].cuda(),
                               return_dict=None, )
                Ap_all_predict += torch.softmax(Apoutput['cls_logits'],dim=-1)[:,1].cpu().tolist()

                cppoutput = cppmodel(input_ids=inputs['input_ids'].cuda(),
                                   feature=inputs['feature'].cuda(),
                                   attention_mask=inputs['attention_mask'].cuda(),
                                   return_dict=None, )
                cpp_all_predict += torch.softmax(cppoutput['cls_logits'], dim=-1)[:, 1].cpu().tolist()

        data['Ap_predict'] = Ap_all_predict
        data['cpp_predict'] = cpp_all_predict
        data[['seq','Ap_predict','cpp_predict']].to_csv(os.path.join(save_path, os.path.split(file)[-1]))
