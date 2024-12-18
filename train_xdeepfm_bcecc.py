# -*- coding: utf-8 -*-
import pandas as pd
import torch
import os
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import xDeepFM_BCECC
from deepctr_torch.logger import setup_logger
from deepctr_torch.callbacks import EarlyStopping
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='my_log_file.txt', help='log place')
parser.add_argument('--gpus', type=int, default=0, help='gpu id')
args = parser.parse_args()

 
if __name__ == "__main__":
    logger = setup_logger(log_file=os.path.join('./logs',args.log))
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
        device = 'cuda:{}'.format(args.gpus)
    else:
        device = 'cpu'  # 添加这行以确保在没有CUDA时代码能正常运行
    es = EarlyStopping(monitor='val_binary_crossentropy', min_delta=0, verbose=1, patience=0, mode='min')
    model = xDeepFM_BCECC(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary_cc',dnn_use_bn=True,dnn_dropout=0.5,
                   l2_reg_embedding=1e-5, device=device)
 
    model.compile("adagrad", "new_bce_cont",
                  metrics=["binary_crossentropy", "auc"], )
 
    history = model.fit(train_model_input, data_train[target].values, batch_size=256, epochs=10, verbose=2,
                        validation_split=0.2, logger=logger, callbacks=[es])
    pred_ans = model.predict(test_model_input, 256)
    logger.info("test LogLoss: {:.4f}".format(log_loss(data_test[target].values, pred_ans)))
    logger.info("test AUC: {:.4f}".format(roc_auc_score(data_test[target].values, pred_ans)))

    