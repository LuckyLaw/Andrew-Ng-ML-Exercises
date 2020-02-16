from __future__ import unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 一元线性回归
learn_rate = 0.01  # 学习速率
train_step = 2000  # 设置迭代次数

onevar_filepath = 'C:/Users/Chen/Desktop/Andrew Ng ML Exercises/Ex1_Linear_Regression/ex1data1.txt'
onevar_data = pd.read_csv(onevar_filepath, header=None, names=['population', 'profit'])
onevar_x = onevar_data['population'].tolist()
onevar_y = onevar_data['profit'].tolist()

# 数据集个数
m = len(onevar_x)

# 一元线性回归代价函数中theta0和theta1初始值
theta0 = tf.Variable(np.random.randn(), name='theta0')
theta1 = tf.Variable(np.random.randn(), name='theta1')


# 目标函数
def linear_function(th0, th1, x):
    h_theta = th0 + th1 * x
    return h_theta


# 代价函数J
def J_function(h_theta, y, m):
    j_function = tf.reduce_sum(tf.pow(h_theta - y, 2)) / (2 * m)  # MSE
    return j_function


# 采用随机梯度下降
optimizer = tf.optimizers.SGD(learn_rate)

# 建立代价函数值list数组
loss = []

# 按照设定步长，执行梯度优化
for step in range(1, train_step + 1):
    with tf.GradientTape() as g:
        pred = linear_function(theta0, theta1, onevar_x)
        cost = J_function(pred, onevar_y, m)

    gradients = g.gradient(cost, [theta0, theta1])
    cost = cost.numpy()
    optimizer.apply_gradients(zip(gradients, [theta0, theta1]))
    # pred = linear_function(theta0, theta1, onevar_x)
    # loss = J_function(pred, onevar_y, m)
    loss.append(cost)
    print(step, cost, theta0.numpy(), theta1.numpy())

# 样本数据与拟合曲线
fig = plt.figure(figsize=(12, 6))
plt.scatter(onevar_x, onevar_y, alpha=0.8, label='whole data set')
plt.plot(onevar_x, np.array(theta1 * onevar_x + theta0), color='tomato', label='Fitted line')
plt.legend()
plt.xlabel('population', fontsize=12)
plt.ylabel('profit', fontsize=12)
plt.grid(True)
plt.show()

# 代价值下降图
step = []
for i in range(0, train_step):
    step.append(i)

fig = plt.figure(figsize=(12, 6))
plt.plot(step, loss, color='tomato', linestyle='--')
plt.legend()
plt.xlabel('population', fontsize=12)
plt.ylabel('profit', fontsize=12)
plt.grid(True)
plt.show()
