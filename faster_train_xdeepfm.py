# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import xDeepFM
import logging
import argparse
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='logs/my_log_file.txt', help='log place')
args = parser.parse_args()

logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(args.log)
file_handler.setLevel(logging.DEBUG)
# 配置日志记录
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
 
# 将文件处理器添加到日志记录器中
logger.addHandler(file_handler)

# 并行化 LabelEncoder
def encode_label(feat):
    lbe = LabelEncoder()
    return feat, lbe.fit_transform(data[feat])

if __name__ == "__main__":
    logger.info("start loading data, which will take around 3 min ...")
    data = pd.read_csv('./data/train.txt', sep='\t')

    logger.info("successfully load data!")
    logger.info("start processing data, which will take around 10 min ...")


    # 定义所有的稀疏特征
    sparse_features = ['C' + str(i) for i in range(1, 27)]

    # 使用 joblib 并行化对每个稀疏特征的 Label Encoding
    encoded_features = Parallel(n_jobs=-1)(delayed(encode_label)(feat) for feat in sparse_features)

    # 更新数据中的稀疏特征
    for feat, encoded in encoded_features:
        data[feat] = encoded

    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = ['label']
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    data[target] = data[target].astype(int)

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name
    max_vals = {feat: data[feat].max() + 1 for feat in sparse_features}

    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=max_vals[feat], embedding_dim=4)
        for feat in sparse_features
    ] + [DenseFeat(feat, 1) for feat in dense_features]
    
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    data_train, data_test = train_test_split(data, test_size=0.2, random_state=2024)
    train_model_input = {name: data_train[name] for name in feature_names}
    test_model_input = {name: data_test[name] for name in feature_names}

    logger.info("successfully get data!")
 
    # 4.Define Model,train,predict and evaluate
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        logger.info('cuda ready...')
        device = 'cuda:7'
    else:
        device = 'cpu'  # 添加这行以确保在没有CUDA时代码能正常运行
 
    model = xDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-5, device=device)
 
    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )
 
    history = model.fit(train_model_input, data_train[target].values, batch_size=64, epochs=10, verbose=2,
                        validation_split=0.2)
    pred_ans = model.predict(test_model_input, 256)
    logger.info("test LogLoss: {:.4f}".format(log_loss(data_test[target].values, pred_ans)))
    logger.info("test AUC: {:.4f}".format(roc_auc_score(data_test[target].values, pred_ans)))

    