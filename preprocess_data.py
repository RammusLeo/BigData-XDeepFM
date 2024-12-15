import pandas as pd

# # 1. 设置列索引
columns_train = ['label\tI1\tI2\tI3\tI4\tI5\tI6\tI7\tI8\tI9\tI10\tI11\tI12\tI13\tC1\tC2\tC3\tC4\tC5\tC6\tC7\tC8\tC9\tC10\tC11\tC12\tC13\tC14\tC15\tC16\tC17\tC18\tC19\tC20\tC21\tC22\tC23\tC24\tC25\tC26']
# columns_test = ['I1\tI2\tI3\tI4\tI5\tI6\tI7\tI8\tI9\tI10\tI11\tI12\tI13\tC1\tC2\tC3\tC4\tC5\tC6\tC7\tC8\tC9\tC10\tC11\tC12\tC13\tC14\tC15\tC16\tC17\tC18\tC19\tC20\tC21\tC22\tC23\tC24\tC25\tC26']

# 2. 读取原始文件内容
with open('./data/train_raw.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 3. 在第一行插入列名
lines.insert(0, '\t'.join(columns_train) + '\n')

# 4. 将修改后的内容写回文件
with open('./data/train.txt', 'w', encoding='utf-8') as f:
    f.writelines(lines)