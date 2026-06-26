import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('..')
from utils import *


class EasyDataset(Dataset):
    def __init__(self, sequences, tokenizer):
        self.length = len(sequences)
        seq = []
        for ss in sequences:
            data_list = re.findall(".{1}", ss)
            seq.append(" ".join(data_list))
        self.all_input = tokenizer.batch_encode_plus(seq, padding=True)

    def __getitem__(self, idx):
        example = {'input_ids':torch.tensor(self.all_input[idx].ids),
                   'attention_mask':torch.tensor(self.all_input[idx].attention_mask)}
        return example

    def __len__(self):
        return self.length


if __name__ == '__main__':
    setup_seed(0)
    file_path = r'D:\siRNA-注意力可视化结果\AMPs-.xlsx'
    seq_columns_name = '序列'  # 列名

    # 创建保存目录（如果不存在）
    AMP_output_dir = r"D:\siRNA-注意力可视化结果\AMP_model_AMPs"
    os.makedirs(AMP_output_dir, exist_ok=True)

    CPP_output_dir = r"D:\siRNA-注意力可视化结果\CPP_model_AMPs"
    os.makedirs(CPP_output_dir, exist_ok=True)

    file_type = os.path.splitext(file_path)[-1]
    if file_type=='.csv':
        df = pd.read_csv(file_path)
    elif file_type=='.xlsx':
        df = pd.read_excel(file_path)
    else:
        raise RuntimeError('请选择csv/excel文件')

    sequence = df[seq_columns_name].values.tolist()

    tokenizer, config = get_TokenizerConfig()
    AMPmodel = load_model(config, use_property=True, CUDA=True)
    AMPmodel.load_state_dict(torch.load('../trained_model/Bert/Ap/Ap_cls_fewfeature_6060train_1fold_0303.pth'))  # 使用哪个模型，可以修改
    CPPmodel = load_model(config, use_property=True, CUDA=True)
    CPPmodel.load_state_dict(torch.load('../trained_model/Bert/cpp/cpp_cls_fewfeature_3936train_3fold_0303.pth'))

    dataset = EasyDataset(sequence, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=128,  # 这个参数是指一次对多少条序列进行注意力可视化
        num_workers=0,  # 这个参数简单来说指采用多少线程来加载处理数据，数值需要根据电脑而定
        pin_memory=True,
        shuffle=False
    )

    with torch.no_grad():
        AMPmodel.eval()
        CPPmodel.eval()
        for i, inputs in enumerate(tqdm(dataloader)):
            AMP_attentions = AMPmodel.get_attention(input_ids=inputs['input_ids'].cuda(),
                                                   attention_mask=inputs['attention_mask'].cuda(),
                                                   return_dict=None)

            CPP_attentions = CPPmodel.get_attention(input_ids=inputs['input_ids'].cuda(),
                                                    attention_mask=inputs['attention_mask'].cuda(),
                                                    return_dict=None)

            batch_size, num_heads, seq_len, _ = AMP_attentions[0].shape
            for layer_idx, (amp_layer_attention, cpp_layer_attention) in enumerate(zip(AMP_attentions, CPP_attentions)):
                for batch_idx in range(batch_size):
                    token = tokenizer.decode(inputs['input_ids'][batch_idx]).split(' ')
                    aa_seq = ''.join(token[1:-1])

                    amp_save_path = os.path.join(AMP_output_dir, f"{aa_seq}")
                    os.makedirs(amp_save_path, exist_ok=True)
                    cpp_save_path = os.path.join(CPP_output_dir, f"{aa_seq}")
                    os.makedirs(cpp_save_path, exist_ok=True)

                    for head_idx in range(num_heads):
                        amp_attention_matrix = amp_layer_attention[batch_idx, head_idx].detach().cpu().numpy()
                        amp_df = pd.DataFrame(amp_attention_matrix,index=token,columns=token)
                        amp_filename = os.path.join(amp_save_path, f"layer_{layer_idx+1}_head_{head_idx + 1}.csv")
                        amp_df.to_csv(amp_filename, float_format="%.4f")

                        cpp_attention_matrix = cpp_layer_attention[batch_idx, head_idx].detach().cpu().numpy()
                        cpp_df = pd.DataFrame(cpp_attention_matrix, index=token, columns=token)
                        cpp_filename = os.path.join(cpp_save_path, f"layer_{layer_idx + 1}_head_{head_idx + 1}.csv")
                        cpp_df.to_csv(cpp_filename, float_format="%.4f")


