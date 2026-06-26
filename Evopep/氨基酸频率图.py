import logomaker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Bio.Data import CodonTable
import Bio
import os

def plot_seqlogo(seqs, letters=None, pos_range=None, seq_len=None, ax=None, color_scheme=None, title=None):
    """
    seqs: list
    """    
    color_scheme = color_scheme if color_scheme is not None else 'chemistry'
    # 删除含有终止子的序列
    seqs = [seq for seq in seqs if '\*' not in seq]
    
    max_len = max([len(seq) for seq in seqs])
    seq_len = max_len if max_len is not None else max([len(seq) for seq in seqs])
    # 删除框架区/酶切.....等
    if pos_range is not None:
        seqs = [seq[:pos_range[0]] + seq[pos_range[0]:pos_range[1]]
                + ''.join(['-'] * (max_len - (pos_range[0] - pos_range[1]) - len(seq[pos_range[0]:pos_range[1]])))
                + seq[pos_range[1]:] for seq in seqs]
    else:
        seqs = [ seq + ''.join(['-']*(max_len-len(seq))) for seq in seqs]
    letters = letters if letters is not None else list(Bio.Data.IUPACData.protein_letters)

    chars = pd.DataFrame( np.array([tuple(seq) for seq in seqs]), columns=list(range(seq_len)))
    pfm = chars.apply(pd.value_counts).T.reindex(columns=letters, fill_value=0.).fillna(0) / len(chars)
    
    logo = logomaker.Logo(pfm, color_scheme=color_scheme, ax=ax, figsize=(15,8))
    if not ax:
        ax = plt.gca()
    ax.axes.get_xaxis().set_visible(True)
    ax.axes.get_yaxis().set_visible(True)
    logo.style_xticks(anchor=0, spacing=1)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel('Position', fontsize=20)
    ax.set_ylabel('Frequency', fontsize=20)
    if title is not None:
        ax.set_title(title)

if __name__ == '__main__':
    # 频率图保存路径(不需要名字)
    pic_save_path = '******'

    # 待统计的文件路径（csv）
    # data_path = './文件路径/文件名字.csv'
    # data_df = pd.read_csv(data_path)

    # 待统计的文件路径（xlsx）
    # data_path = './文件路径/文件名字.xlsx'
    # data_df = pd.read_excel(data_path)

    for length in set(data_df['seq'].str.len().tolist()):
        seq = data_df[data_df['seq'].str.len()==length]['seq'].tolist()
    	plot_seqlogo(seqs)
	plt.savefig(os.path.join(pic_save_path, f'{os.path.splitext(os.path.split('./data/skempi_sabdab.csv')[-1])[0]}_length_{length}_frequency_pic.png'))	
    	plt.show()