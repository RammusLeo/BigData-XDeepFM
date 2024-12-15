import pandas as pd
import numpy as np

train_data = pd.read_csv('./data/train.txt', sep='\t')
datalen = len(train_data)
sample_len = datalen // 10
train_data = train_data[:sample_len]
train_data.to_csv('./data/train_sample_10p.txt', sep='\t', index=False)