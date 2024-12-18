# -*- coding: utf-8 -*-
import pandas as pd
import torch
import os
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import xDeepFM_BCECC
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='my_log_file.txt', help='log place')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
args = parser.parse_args()

logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join('./logs',args.log))
file_handler.setLevel(logging.DEBUG)
# 配置日志记录
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
 
# 将文件处理器添加到日志记录器中
logger.addHandler(file_handler)

if __name__ == "__main__":
    logger.info("start loading data, which will take around 3 min ...")
    data = pd.read_csv('./data/train_sample_10p.txt', sep='\t')
    # data_test = pd.read_csv('./data/test.txt')
    logger.info("successfully load data!")
    logger.info("start processing data, which will take around 10 min ...")

    sparse_features = ['C' + str(i) for i in range(1, 27)]
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

    # fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
    #                           for feat in sparse_features] + [DenseFeat(feat, 1, )
    #                                                           for feat in dense_features]
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
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'  # 添加这行以确保在没有CUDA时代码能正常运行
 
    model = xDeepFM_BCECC(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary_cc',
                   l2_reg_embedding=1e-5, device=device)
 
    model.compile("adagrad", "new_bce_cont",
                  metrics=["binary_crossentropy", "auc"], )
 
    history = model.fit(train_model_input, data_train[target].values, batch_size=256, epochs=10, verbose=2,
                        validation_split=0.2)
    pred_ans = model.predict(test_model_input, 256)
    logger.info("test LogLoss: {:.4f}".format(log_loss(data_test[target].values, pred_ans)))
    logger.info("test AUC: {:.4f}".format(roc_auc_score(data_test[target].values, pred_ans)))

    