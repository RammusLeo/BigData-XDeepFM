# SLFF-xDeepFM
## 李奡 蒋浩楠
### Training
1. run `preprocess_data.py` to label the indices.
2. run `extract_train_sample.py` to split data for experiments.
3. run `train_xdeepfm.py`
```
python train_xdeepfm_bcecc.py --log my_log_file.txt --gpus 7
```
4. logs are saved in `/logs`, however the training progress is not saved, which needs tensorboard or something to record.
