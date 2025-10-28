import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Bidirectional
# import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_squared_error
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import Sequential, layers, utils
from tensorflow.keras.optimizers import Adam

start_time = time.time()
dataset = pd.read_csv('cpu.csv')
# 显示shape
dataset.shape
# 默认显示前5行
dataset.head()
print(dataset.head())
# 显示数据描述
dataset.describe()
# 将字段Datetime数据类型转换为日期类型
dataset['Datetime'] = pd.to_datetime(dataset['Datetime'], format="%Y-%m-%d %H:%M:%S")
# 将字段Datetime设置为索引列
# 目的：后续基于索引来进行数据集的切分
dataset.index = dataset.Datetime
# 将原始的Datetime字段列删除
dataset.drop(columns=['Datetime'], axis=1, inplace=True)
# 显示默认前5行
dataset.head()
# 可视化显示DOM_MW的数据分布情况

# dataset['DOM_MW'].plot(figsize=(16,8))
# plt.show()
# 数据进行归一化
# 均值为0，标准差为1
scaler = MinMaxScaler()
# reshape(-1, 1) 第一个-1不管多少行，第二个1只是1列
dataset['Value'] = scaler.fit_transform(dataset['Value'].values.reshape(-1, 1))
# 可视化显示归一化后的数据分布情况

# dataset['DOM_MW'].plot(figsize=(16,8))
# plt.show()


# 功能函数：构造特征数据集和标签集
def create_new_dataset(dataset, seq_len=12):
    '''基于原始数据集构造新的序列特征数据集
    Params:
        dataset : 原始数据集
        seq_len : 序列长度（时间跨度） 滑动窗口

    Returns:
        X, y
    '''
    X = []  # 初始特征数据集为空列表
    y = []  # 初始标签数据集为空列表

    start = 0  # 初始位置
    end = dataset.shape[0] - seq_len  # 截止位置

    for i in range(start, end):  # for循环构造特征数据集
        sample = dataset[i: i + seq_len]  # 基于时间跨度seq_len创建样本
        label = dataset[i + seq_len]  # 创建sample对应的标签
        X.append(sample)  # 保存sample
        y.append(label)  # 保存label

    # 返回特征数据集和标签集
    return np.array(X), np.array(y)


# 功能函数：基于新的特征的数据集和标签集，切分：X_train, X_test
# 千万不能打乱数据 要有时序
def split_dataset(X, y, train_ratio=0.8):
    '''基于X和y，切分为train和test
    Params:
        X : 特征数据集
        y : 标签数据集
        train_ratio : 训练集占X的比例

    Returns:
        X_train, X_test, y_train, y_test
    '''
    X_len = len(X)  # 特征数据集X的样本数量
    train_data_len = int(X_len * train_ratio)  # 训练集的样本数量

    X_train = X[:train_data_len]  # 训练集
    y_train = y[:train_data_len]  # 训练标签集

    X_test = X[train_data_len:]  # 测试集
    y_test = y[train_data_len:]  # 测试集标签集

    # 返回值
    return X_train, X_test, y_train, y_test


# 功能函数：基于新的X_train, X_test, y_train, y_test创建批数据(batch dataset)
def create_batch_data(X, y, batch_size=32, data_type=1):
    '''基于训练集和测试集，创建批数据
    Params:
        X : 特征数据集
        y : 标签数据集
        batch_size : batch的大小，即一个数据块里面有几个样本
        data_type : 数据集类型（测试集表示1，训练集表示2）

    Returns:
        train_batch_data 或 test_batch_data
    '''
    if data_type == 1:  # 测试集
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))  # 封装X和y，成为tensor类型
        test_batch_data = dataset.batch(batch_size)  # 构造批数据
        # 返回
        return test_batch_data
    else:  # 训练集
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))  # 封装X和y，一一对应，成为tensor类型
        # 训练集数据量较大，可以加载到内存中去 打乱1000 获得更好地泛化性能
        train_batch_data = dataset.cache().shuffle(1000).batch(batch_size)  # 构造批数据
        # 返回
        return train_batch_data

# ① 原始数据集
dataset_original = dataset
# print(dataset_original)

# ② 构造特征数据集和标签集，seq_len序列长度为12小时
SEQ_LEN = 12 # 序列长度
X, y = create_new_dataset(dataset_original.values, seq_len = SEQ_LEN)

# ③ 数据集切分
X_train, X_test, y_train, y_test = split_dataset(X, y, train_ratio=0.9)

# ④ 基于新的X_train, X_test, y_train, y_test创建批数据(batch dataset)
# 测试批数据
test_batch_dataset = create_batch_data(X_test, y_test, batch_size=256, data_type=1)

# 训练批数据
train_batch_dataset = create_batch_data(X_train, y_train, batch_size=256, data_type=2)

# 假设 SEQ_LEN 的值已经定义
SEQ_LEN = 12

# 创建一个手动实现的双向 GRU 类
class BidirectionalGRU(layers.Layer):
    def __init__(self, units):
        super(BidirectionalGRU, self).__init__()
        self.forward_layer = layers.GRU(units, return_sequences=True)
        self.backward_layer = layers.GRU(units, return_sequences=True, go_backwards=True)

    def call(self, inputs):
        forward_output = self.forward_layer(inputs)
        backward_output = self.backward_layer(inputs)
        # 保留最后一个时间步的输出
        forward_output_last = forward_output[:, -1, :]
        backward_output_last = backward_output[:, 0, :]  # 反向传播的最后一个时间步
        return layers.concatenate([forward_output_last, backward_output_last])

model = Sequential([
    BidirectionalGRU(16),
    layers.Dense(1)
])

# 定义 checkpoint，保存权重文件
file_path = "best_checkpoint.hdf5"
# 最小损失权重
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                         monitor='loss',
                                                         mode='min',
                                                         save_best_only=True,
                                                         save_weights_only=True)
# 模型编译
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='mse',
    metrics=['mae']
)

# 模型训练
history = model.fit(train_batch_dataset,
          epochs=5,
          validation_data=test_batch_dataset,
          callbacks=[checkpoint_callback])

# 显示 train loss 和 val loss
plt.figure(figsize=(16,8))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title("LOSS")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc='best')
plt.show()
# 模型验证
test_pred = model.predict(X_test, verbose=1)

# 计算r2
score = r2_score(y_test, test_pred)
print("r^2 的值： ", score)
# 计算均方误差 (MSE)
mse = mean_squared_error(y_test, test_pred)
print("均方误差 (MSE): ", mse)
# 计算均方根误差 (RMSE)
rmse = mean_squared_error(y_test, test_pred, squared=False)
print("均方根误差 (RMSE): ", rmse)
# 计算平均绝对误差 (MAE)
mae = mean_absolute_error(y_test, test_pred)
print("平均绝对误差 (MAE): ", mae)

# 绘制模型验证结果
plt.figure(figsize=(16,8))
plt.plot(y_test, label="True label")
plt.plot(test_pred, label="Pred label")
plt.title("True vs Pred")
plt.legend(loc='best')
plt.show()

end_time = time.time()
print("Bi-GRU运行时间：", end_time-start_time)
