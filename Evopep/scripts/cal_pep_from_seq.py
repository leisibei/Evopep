# -*- coding: utf-8 -*-
# from . import BasicDes
import BasicDes, Autocorrelation, CTD, PseudoAAC, AAComposition, QuasiSequenceOrder
import pandas as pd
import numpy as np
import sys
import multiprocessing
"""
直接给list形式的氨基酸序列生成
"""
def cal_pep(peptide):

    peptide = str(peptide)
    peptides_descriptor={}
    AAC = AAComposition.CalculateAAComposition(peptide)
    DIP = AAComposition.CalculateDipeptideComposition(peptide)
    MBA = Autocorrelation.CalculateNormalizedMoreauBrotoAutoTotal(peptide, lamba=5)
    CCTD = CTD.CalculateCTD(peptide)
    QSO = QuasiSequenceOrder.GetSequenceOrderCouplingNumberTotal(peptide, maxlag=5)
    PAAC = PseudoAAC._GetPseudoAAC(peptide,lamda=5)
    APAAC = PseudoAAC.GetAPseudoAAC(peptide, lamda=5)
    Basic = BasicDes.cal_discriptors(peptide)
    peptides_descriptor.update(AAC)
    peptides_descriptor.update(DIP)
    peptides_descriptor.update(MBA)
    peptides_descriptor.update(CCTD)
    peptides_descriptor.update(QSO)
    peptides_descriptor.update(PAAC)
    peptides_descriptor.update(APAAC)
    peptides_descriptor.update(Basic)

    return peptides_descriptor


if __name__ == "__main__":
    from tqdm import tqdm
    file = r'E:\peptide-cpy\result\2025.12.17-try-batch\rank\try-pep.csv' # 待计算table特征的文件(之前是#E:\LS\深度学习\code\code_new_0815\practice-cxy\排序后\rank_result_10-20_1000_20250708.csv)
    output_path = r'E:\peptide-cpy\result\2025.12.17-try-batch\rank\descriptor-pep.csv'  # 保存路径以及文件名字（之前是E:\LS\深度学习\code\code_new_0815\practice-cxy\排序后\rank_result_10to20_1000_descriptor_cxy.csv）
    data = pd.read_csv(file, index_col=0)
    data = data.reset_index(drop=True)
    data = data[['seq', 'Ap_predict', 'cpp_predict']]  # 如果有模型预测数值，就用这行,列名可以修改
    # data = data[['seq']]  # 如果只有序列就用这行
    data = data.rename(columns={'seq':'sequence'})
    sequence = data["sequence"]
    peptides = sequence.values.copy().tolist()
    
    result = []
    for seq in tqdm(peptides):
        result_dict = cal_pep(seq)
        result_dict.update({"sequence":seq})
        result.append(result_dict)
    pep_df = pd.DataFrame(result)
    
    data_df = pd.merge(data,pep_df,how='inner',on='sequence')
    data_df.to_csv(output_path)
