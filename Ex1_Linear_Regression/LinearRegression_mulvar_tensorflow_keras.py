from __future__ import unicode_literals
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 数据读取
filePath = 'ex1data2.txt'
ex1Data2 = pd.read_csv(filePath, header=None, names=['size', 'no_bedrooms', 'price'])

# 数据特征缩放
x_size_mean = np.mean(ex1Data2.iloc[:, 0])
x_no_bedrooms_mean = np.mean(ex1Data2.iloc[:, 1])
x_size_std = np.std(ex1Data2.iloc[:, 0])
x_no_bedrooms_std = np.std(ex1Data2.iloc[:, 1])
y_mean = np.mean(ex1Data2.iloc[:, -1])
y_std = np.std(ex1Data2.iloc[:, -1])

ex1Data2.iloc[:, 0] = (ex1Data2.iloc[:, 0] - x_size_mean) / x_size_std  # size特征缩放
ex1Data2.iloc[:, 1] = (ex1Data2.iloc[:, 1] - x_no_bedrooms_mean) / x_no_bedrooms_std  # no_bedrooms特征缩放
ex1Data2.iloc[:, -1] = (ex1Data2.iloc[:, -1] - y_mean) / y_std
x_input = ex1Data2.iloc[:, 0:-1]
y_input = ex1Data2.iloc[:, -1]

# 可变参数设置
alpha = 0.001  # 学习速率
train_step = 2000  # 步长

# 建立神经网络模型
model = tf.keras.Sequential()  # 向前传递
model.add(tf.keras.layers.Dense(1, input_shape=(2,)))  # 一层网络神经，输入两个x值
model.summary()  # 模型展示
print(model.summary())

# 选择优化方法
optimizer = tf.keras.optimizers.SGD(alpha)

# 选择损失函数
j_function = tf.keras.losses.MeanSquaredError()

# 模型配置
model.compile(
    optimizer=optimizer,
    loss=j_function,
    metrics=['accuracy']
)

# 建立tensorboard监听
log_dir = "logs\\fit_mulvar\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 模型编译与训练
linear_regression_result = model.fit(x=x_input,
                                     y=y_input,
                                     epochs=train_step,
                                     validation_split=0.1,
                                     callbacks=[tensorboard_callback]
                                     )

# 输入经过拟合曲线获得输出
y_fit = model.predict(x_input)
data_no = np.arange(47)
w, b = model.layers[0].get_weights()  # 获取权重和偏差
print(w)
print(b)

# 拟合后输出与原输出做对比
fig = plt.figure(figsize=(12, 6))
plt.scatter(data_no, y_input.tolist(), alpha=0.8, label='y_dataset')
plt.plot(data_no, y_fit[:, 0], color='tomato', label='y_fit')
plt.legend()
plt.xlabel('number of data', fontsize=12)
plt.ylabel('y_output', fontsize=12)
plt.grid(True)
plt.show()

# 输出拟合的目标函数
print('The objective function is: h_theta=%0.4f+%0.4f*x1+%0.4f*x2' % (b, w[0], w[1]))
