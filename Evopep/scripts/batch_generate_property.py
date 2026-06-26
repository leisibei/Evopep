import pandas as pd
import os
import sys

sys.path.append('..')
from generate_feature import Batch_BasicDes


def GenerateSample(csv_path, save_path):
    all_sample = pd.read_csv(csv_path, index_col=0)
    sequence = all_sample["seq"]
    peptides = sequence.values.copy().tolist()
    save_path = os.path.join(save_path, os.path.split(csv_path)[-1])
    Batch_BasicDes.cal_pep(peptides, sequence, save_path)


if __name__ == "__main__":
    # 待筛选文件所在的文件夹，需要修改,末尾有无\都可以
    data_path = r'E:\LS\深度学习\code\code_new_0815\practice-cxy\input\经验筛选'

    # 生成理化性质后保存的路径，需要修改，末尾有无\都可以,这个文件夹可以不用手动创建，如果没有这个文件夹，代码会生成
    save_path = r'E:\LS\深度学习\code\code_new_0815\practice-cxy\理化性质'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file_name in os.listdir(data_path):
        if os.path.splitext(file_name)[-1] == '.csv':
            print(file_name)
            GenerateSample(csv_path=os.path.join(data_path, file_name), save_path=save_path)
        else:
            continue
