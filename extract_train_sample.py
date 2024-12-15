import pandas as pd
import numpy as np

train_data = pd.read_csv('./data/train.txt', sep='\t')
train_data = train_data[:100]
train_data.to_csv('./data/train_sample.txt', sep='\t', index=False)