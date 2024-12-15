# xDeepFM
### Training
1. run `preprocess_data.py` to label the indices.
2. run `extract_train_sample.py` to split 10% data for experiments.
3. run `train_xdeepfm.py` (faster的还不能用 也不fast)
```
python train_xdeepfm --log my_log_file.txt --gpu 0
```
4. logs are saved in `/logs`, however the training progress is not saved, which needs tensorboard or something to record.