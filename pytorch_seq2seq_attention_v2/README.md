#### 目录结构
```
├── checkpoints/
├── dataset/
│   ├── __init__.py
│   ├── utils.py
│   └── *.csv
├── models/
│   ├── __init__.py
│   └── Seq2SeqAttention.py
└── utils.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
```

其中：

- checkpoints/： 用于保存训练好的模型，可使程序在异常退出后仍能重新载入模型，恢复训练
- datasets/：数据集以及数据相关操作，包括数据预处理、dataset实现等
- models/：模型定义
- utils/：可能用到的工具函数
- train.py：训练和程序的入口
- predict.py：模型预测文件
- requirements.txt：程序依赖的第三方库
- README.md：程序的必要说明
- pytorch_seq2seq_attention_v2.ipynb：Jupyter Notebook 文件，用于演示


#### 安装环境
```
pip install -r requirements.txt
```

#### 训练

```
python train.py
```

```
Load Datasets
Split train and valid datasets
training...
700it [00:57, 12.20it/s]
evaluating...
300it [00:10, 27.57it/s]
| Epoch: 001 | Train Loss: 1.736 | Train PPL:   5.673 | Val. Loss: 0.510 | Val. PPL:   1.665|
training...
700it [01:03, 11.05it/s]
evaluating...
300it [00:10, 28.25it/s]
| Epoch: 002 | Train Loss: 0.641 | Train PPL:   1.899 | Val. Loss: 0.267 | Val. PPL:   1.306|
training...
700it [01:04, 10.88it/s]
evaluating...
300it [00:11, 27.07it/s]
| Epoch: 003 | Train Loss: 0.331 | Train PPL:   1.392 | Val. Loss: 0.052 | Val. PPL:   1.054|
```


#### 预测

```
python predict.py
```

```
load model from checkpoints...
model summary:
Seq2Seq(
  (encoder): Encoder(
    (embedding): Embedding(13, 8)
    (rnn): GRU(8, 32, bidirectional=True)
    (fc): Linear(in_features=64, out_features=32, bias=True)
    (dropout): Dropout(p=0.5)
  )
  (decoder): Decoder(
    (attention): Attention(
      (attn): Linear(in_features=96, out_features=32, bias=True)
    )
    (embedding): Embedding(13, 8)
    (rnn): GRU(72, 32)
    (out): Linear(in_features=104, out_features=13, bias=True)
    (dropout): Dropout(p=0.5)
  )
)
load test inputs dataset:
predicting...
1000it [00:35, 28.02it/s]
Generating reversed sequence csv done!
```


#### 结果文件
```
reverse_seq_predictions.csv
```