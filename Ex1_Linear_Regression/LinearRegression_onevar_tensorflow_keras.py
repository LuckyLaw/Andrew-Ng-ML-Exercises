from __future__ import unicode_literals
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 数据读取
filePath = 'C:/Users/Chen/Desktop/Andrew_Ng_ML_Exercises/Ex1_Linear_Regression/ex1data1.txt'
ex1Data = pd.read_csv(filePath, header=None, names=['population', 'profit'])
x_onevar = ex1Data['population']
y_onevar = ex1Data['profit']

# 常量设置
alpha = 0.001  # 学习速率
train_step = 3000  # 迭代次数

# 建立神经网络顺序模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))  # Dense可理解为W*X+b,1为输出维度,input_shape是指输入张量，wx+b为一维张量，故输入1
model.summary()
print(model.summary())  # 查看模型

# 选择优化方法
optimizer = tf.keras.optimizers.SGD(alpha)  # 随机梯度下降

# 选择代价函数
j_function = tf.keras.losses.MeanSquaredError()  # MeanSquareError，均方差函数

# 模型编译
model.compile(
    optimizer=optimizer,
    loss=j_function,
)

# 建立tensorboard监听
log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 拟合结果
linear_regression_result = model.fit(x_onevar,  # 输入x值
                                     y_onevar,  # 输入y值
                                     epochs=train_step,  # 输入步长
                                     validation_split=0.1,
                                     callbacks=[tensorboard_callback])  # 导入监听

# 查看模型
q = model.predict(x_onevar)

# 样本数据与拟合曲线
fig = plt.figure(figsize=(12, 6))
plt.scatter(x_onevar, y_onevar, alpha=0.8, label='whole data set')
plt.plot(x_onevar, q[:, 0], color='tomato', label='Fitted line')
plt.legend()
plt.xlabel('population', fontsize=12)
plt.ylabel('profit', fontsize=12)
plt.grid(True)
plt.show()
