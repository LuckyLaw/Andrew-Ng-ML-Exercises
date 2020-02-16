from __future__ import unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 多元线性回归
learn_rate = 0.01  # 学习速率
train_step = 2000  # 设置迭代次数

mulvar_filepath = 'C:/Users/Chen/Desktop/Andrew Ng ML Exercises/Ex1_Linear_Regression/ex1data2.txt'
mulvar_data = pd.read_csv(mulvar_filepath, header=None, names=['size', 'no_bedrooms', 'price'])
mulvar_x1 = mulvar_data['size'].tolist()
mulvar_x2 = mulvar_data['no_bedrooms'].tolist()
mulvar_y = mulvar_data['price'].tolist()

# 特征缩放
x1_mean = np.mean(mulvar_x1)
x2_mean = np.mean(mulvar_x2)
y_mean = np.mean(mulvar_y)
x1_std = np.std(mulvar_x1)
x2_std = np.std(mulvar_x2)
y_std = np.std(mulvar_y)
x11 = []
x21 = []
y1 = []

for i in range(len(mulvar_y)):
    x11.append((mulvar_x1[i] - x1_mean) / x1_std)
    x21.append((mulvar_x2[i] - x2_mean) / x1_std)
    y1.append((mulvar_y[i] - y_mean) / y_std)

# print(x1)
# print(x2_mean)


# 数据集个数
m = len(mulvar_x1)

# 一元线性回归代价函数中theta0、theta1和theta2初始值
theta0 = tf.Variable(np.random.randn(), name='theta0')
theta1 = tf.Variable(np.random.randn(), name='theta1')
theta2 = tf.Variable(np.random.randn(), name='theta2')

print(mulvar_x1)
print(mulvar_x2)
print(mulvar_y)


# 目标函数
def h_function(th0, th1, th2, x1, x2):
    h_theta = th0 + th1 * x1 + th2 * x2
    return h_theta


def J_function(h_function, y, m):
    j_function = tf.reduce_sum(tf.pow(h_function - y, 2)) / (2 * m)  # MSE
    return j_function


# 采用随机梯度法下降
optimizer = tf.optimizers.SGD(learn_rate)

# 建立代价函数值list数组
loss = []

# 按照设定步长，执行梯度优化
for step in range(1, train_step + 1):
    with tf.GradientTape() as g:
        pred = h_function(theta0, theta1, theta2, x11, x21)
        cost = J_function(pred, y1, m)

    gradients = g.gradient(cost, [theta0, theta1, theta2])
    optimizer.apply_gradients(zip(gradients, [theta0, theta1, theta2]))
    cost = cost.numpy()
    loss.append(cost)
    print(step, cost, theta0.numpy(), theta1.numpy(), theta2.numpy())

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
print('The objective function is: h_theta=%0.4f+%0.4f*x1+%0.4f*x2' % (theta0.numpy(), theta1.numpy(), theta2.numpy()))
