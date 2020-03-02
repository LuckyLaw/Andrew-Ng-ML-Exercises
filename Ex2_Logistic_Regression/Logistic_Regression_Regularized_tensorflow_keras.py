from __future__ import unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

# 读取数据
filePath = 'ex2data2.txt'
ex2data2 = pd.read_csv(filePath, header=None)
x1 = ex2data2.iloc[:, 0]
x2 = ex2data2.iloc[:, 1]
y = ex2data2.iloc[:, -1]

# 正则化参数和步长
lambda_input = 0.01  # 正则化参数
train_step = 1500  # 步长


# 建立特征
def feature_mapping(x1, x2, power):
    data = {}
    for i in np.arange(power + 1):
        for p in np.arange(i + 1):
            data["f{}{}".format(i - p, p)] = np.power(x1, i - p) * np.power(x2, p)

            #     data = {"f{}{}".format(i - p, p): np.power(x1, i - p) * np.power(x2, p)
            #                 for i in np.arange(power + 1)
            #                 for p in np.arange(i + 1)
            #             }
    return pd.DataFrame(data)


mapFeature = feature_mapping(x1, x2, power=6).drop('f00', axis=1)  # 第一列值都为1，所以删掉第一列

# keras建立神经网络
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(27,), kernel_regularizer=tf.keras.regularizers.l2(lambda_input),
                                activation='sigmoid'))  # keras.regularizers.l2为权重平方和
# model.add(tf.keras.layers.Dense(5, kernel_regularizer=tf.keras.regularizers.l2(0.001),
#                                 activation='relu'))  # keras.regularizers.l2为权重平方和
# model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.001),
#                                 activation='sigmoid'))  # keras.regularizers.l2为权重平方和

model.summary()

# 编译模型
model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=['acc']
)

# 导入监听
log_dir = "logs\\fit_binary_2\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 模型训练
logistic_regression_regularized = model.fit(x=mapFeature,
                                            y=y,
                                            epochs=train_step,
                                            callbacks=[tensorboard_callback]
                                            )

# 输出权重和偏差
w, b = model.layers[0].get_weights()
w = w.tolist()  # 权重
b = b.tolist()  # 偏差

theta = []
theta.append(b[0])

for i in range(0, len(w)):
    theta.append(w[i][0])
theta = np.array(theta)  # 包含偏差和权重的数组，偏差在第一个位置

# 用等高线法将决策边界画出来
x = np.linspace(-1, 1.5, 250)
xx, yy = np.meshgrid(x, x)
z = feature_mapping(xx.ravel(), yy.ravel(), 6).values
z = z @ theta
z = z.reshape(xx.shape)
fig, ax = plt.subplots(figsize=(12, 6))
plt.scatter(x1, x2, alpha=0.8, c=y.tolist())
plt.contour(xx, yy, z, 0)  # 等高线
plt.ylim(-.8, 1.2)
plt.show()
