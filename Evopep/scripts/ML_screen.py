# -*- coding: utf-8 -*-
import joblib
import glob
import pandas as pd
import numpy as np

# +
class_name = 'siRNA'  # 只能是siRNA或者mRNA
df = pd.read_csv(r'E:\LS\深度学习\code\code_new_0815\practice-cxy\排序后\rank_result_10to20_1000_descriptor_cxy.csv',
                 index_col=0)  # 待筛选文件，文件必须包含table中的所有特征， 可以用generate_feature文件夹中的cal_pep_from_seq.py计算，特征一共有676列
feature_df = df.iloc[:, 3:]  # 如果只有序列且序列在第一列，没有模型预测值，那就df.iloc[:, 1:],如果有两个模型的预测值且序列以及预测值在前三列，那就df.iloc[:, 3:]
print('特征维度为：', feature_df.shape)  # 应该输出(样本数量,676),其中676是特征数量
model_path_list = glob.glob(f'../FewSamplesResult/{class_name}/model/*.pkl')

model_list = []
for model_path in model_path_list:
    with open(model_path, 'rb') as f:
        model_list.append(joblib.load(f))
# -

pred_y = np.array([model.predict_proba(feature_df)[:, 1] for model in model_list])
pred_y = pred_y.transpose()
pred_y_all = pred_y.sum(axis=1)

pred_per = pd.DataFrame(pred_y, columns=range(1, len(model_list)+1))
pred_per

pred_df = df.iloc[:, :3]
pred_df[f'predict_{class_name}等级'] = pred_y_all
pred_df = pd.concat([pred_df, pred_per], axis=1)
pred_df

pred_df.to_csv(f'E:/LS/深度学习/code/code_new_0815/practice-cxy/排序后/Final_10to20_ml_result_1000_20250708_cxy.csv')  # 保存路径
