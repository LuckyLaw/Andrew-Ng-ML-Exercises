from __future__ import unicode_literals
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 数据读取
filePath = 'ex2data1.txt'
ex2Data = pd.read_csv(filePath, header=None)
x = ex2Data.iloc[:, :-1]  # 选取前两列
y = ex2Data.iloc[:, -1]  # 选取第三列，也就是最后一列
x1 = ex2Data.iloc[:, 0].tolist()
x2 = ex2Data.iloc[:, 1].tolist()

# Keras建立神经网络
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(2,), activation='sigmoid'))  # sigmoid函数表达式为g(z)=1/(1+e^-z)
model.summary()

# 模型编译
model.compile(
    optimizer='adam',  # 优化方法
    loss='binary_crossentropy',  # 代价函数为交叉熵
    metrics=['acc']
)

# 导入监听
log_dir = "logs\\fit_binary\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 模型训练
logistic_regression_result = model.fit(x=x,
                                       y=y,
                                       epochs=5000,
                                       callbacks=[tensorboard_callback]
                                       )

# 输入经过拟合曲线获得输出
y_fit = model.predict(x)
w, b = model.layers[0].get_weights()  # 输出权重和偏差

# 画分类曲线
xx = np.arange(0, max(x1), 0.1)
yy = -(b / w[1] + (w[0] / w[1]) * xx)

data_no = np.arange(len(y))

# 拟合后输出与原输出做对比
fig = plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(data_no, y.tolist(), alpha=0.8, label='y_dataset')
plt.scatter(data_no, y_fit[:, 0], color='tomato', label='y_fit')
plt.axhline(y=0.5, ls="--", c="orange")  # 添加水平直线
plt.legend()
plt.xlabel('number of data', fontsize=12)
plt.ylabel('y_output', fontsize=12)
plt.grid(True)

plt.subplot(122)
plt.scatter(x1, x2, alpha=0.8, c=y.tolist())
plt.plot(xx, yy, color='tomato', label='Fitted line')
plt.legend()
plt.xlabel('exam1 score', fontsize=12)
plt.ylabel('exam2 score', fontsize=12)
plt.xlim(25,100)
plt.ylim(25,100)
plt.grid(True)
plt.show()
