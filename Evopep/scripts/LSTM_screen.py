import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

sys.path.append('..')
from model import LstmNet
# from utils import *
from dataset import LoadDataset
from dataset import DataCollator


if __name__ == '__main__':
    AMP_model = torch.load('../trained_model/LSTM/AMP/fold1_20230814.pt')  # 使用哪个模型，可以修改
    CPP_model = torch.load('../trained_model/LSTM/CPP/fold1_20230814.pt')  # 使用哪个模型，可以修改
    AMP_model = AMP_model.cuda()
    CPP_model = CPP_model.cuda()

    # 待筛选文件所在文件夹，需要修改，末尾有无\都可以, 此时的文件应该是包括氨基酸序列，多种基本的理化性质,*.csv代表这个路径下所有的csv文件
    library_path = r'E:\LS\深度学习\代码\code_new_0815\practice\1_生成理化性质后文件夹/*.csv'

    # 运行完后，结果保存的文件夹，这个文件夹可以不用手动创建，如果没有这个文件夹，代码会生成, 需要修改，末尾有无\都可以
    save_path = r'E:\LS\深度学习\代码\code_new_0815\practice\2_AI预测可能性文件夹\LSTM'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for csv_name in glob(library_path):
        file = os.path.join(library_path, csv_name)
        print(file)
        data = pd.read_csv(file, index_col=0)
        mydataset = LoadDataset.lstm_dataset(data, use_property=True)
        mydataloader = DataLoader(
            mydataset,
            batch_size=256,  # 这个参数是指一次对多少条序列进行评分，数值需要根据电脑而定，
            num_workers=8,  # 这个参数简单来说指采用多少线程来加载处理数据，数值需要根据电脑而定
            pin_memory=True,
            shuffle=False,
            collate_fn=DataCollator.lstm_collate_fn,
        )
        Ap_all_predict = []
        cpp_all_predict = []
        with torch.no_grad():
            AMP_model.eval()
            CPP_model.eval()
            for i, (x, feature, y) in enumerate(tqdm(mydataloader)):
                Apoutput = AMP_model(x.cuda(), feature.cuda())
                Ap_all_predict += torch.softmax(Apoutput, dim=-1)[:, 1].cpu().tolist()

                cppoutput = CPP_model(x.cuda(), feature.cuda())
                cpp_all_predict += torch.softmax(cppoutput, dim=-1)[:, 1].cpu().tolist()

        data['Ap_predict'] = Ap_all_predict
        data['cpp_predict'] = cpp_all_predict
        data[['seq','Ap_predict','cpp_predict']].to_csv(os.path.join(save_path, os.path.split(file)[-1]))
