# -*- coding: utf-8 -*-

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import load_model
from SimpleData import load_data, load_data_test

# 加载数据
data, label = load_data()
label = np_utils.to_categorical(label, 10)

datatest, labeltest = load_data_test()
labeltest = np_utils.to_categorical(labeltest, 10)

# 构建模型
model = Sequential()
# 第一层为二维卷积层
# 32 为filters卷积核的数目，也为输出的维度
# kernel_size 卷积核的大小，3x3
# 激活函数选为relu 
# 第一层必须包含输入数据规模input_shape这一参数，后续层不必包含
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
# 再加一层卷积，64个卷积核
model.add(Conv2D(64, (3, 3), activation='relu'))
# 加最大值池化
model.add(MaxPooling2D(pool_size=(2, 2)))
# 加Dropout，断开神经元比例为25%
model.add(Dropout(0.25))
# 加Flatten，数据一维化
model.add(Flatten())
# 加Dense，输出128维
model.add(Dense(128, activation='relu'))
# 再一次Dropout
model.add(Dropout(0.5))
# 最后一层为Softmax，输出为10个分类的概率
model.add(Dense(10, activation='softmax'))

model.add(Dense(10, ))

# 优化设置
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# 配置模型，损失函数采用交叉熵，优化采用Adadelta，将识别准确率作为模型评估
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(data, label, epochs=20, batch_size=100)

# 开始评估模型效果
score = model.evaluate(datatest, labeltest, batch_size=10)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 保存模型
model.save('trial_model.h5') 
del model

# 打开模型
model = load_model('trial_model.h5')
# 开始评估模型效果
score = model.evaluate(datatest, labeltest, batch_size=10)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
