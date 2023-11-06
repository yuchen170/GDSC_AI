# GDSC_AI

## Step 0
* 註冊&登入 Google Colab (共編、不用設定環境、免費！)
  [Google Colab](https://colab.research.google.com/)
* 註冊&登入 Kaggle (很多資料集跟比賽可以練功、免費！)
  [Kaggle](https://www.kaggle.com/)

## Step 1.1 import libraries
```
import matplotlib.pyplot as plt # 畫圖用的
import tensorflow as tf # tensorflow!
import pandas as pd # 資料處理&分析的好工具
import numpy as np # 陣列運算、很多的數學函數
```
## Step 1.2 串接Kaggle API 拿資料集

* 登入Kaggle => 點右上角個人頭像 => Your Profile => Account => API => Create New Token
* 下載到桌面
* Paste following code

## Step 2 觀察資料集

```
print(data.info()) # 簡單的訊息
print(data.head()) # 前五筆資料

class_counts = data['label'].value_counts()
print(class_counts) # 各類別的資料筆數
```

## Step 3 切分資料集
```
# 比例可以調整
train_data = data.head(int(len(reduced_data)*0.8))
val_data = data.tail(int(len(reduced_data)*0.2)) 

train_labels = train_data.pop('label')
val_labels = val_data.pop('label')

tf_train_data = tf.data.Dataset.from_tensor_slices((train_data.values, train_labels.values))
tf_val_data = tf.data.Dataset.from_tensor_slices((val_data.values, val_labels.values))
```

## Step 4
