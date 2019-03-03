# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 00:37:33 2019

@author: asus-pc
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(20, activation=tf.nn.relu),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(20, activation=tf.nn.relu),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

predictions = model.predict(x_test)

#Проверяем номера тестов на которых стоят пятерки
def Num_Index(n):
    for i in range(10000):
        if y_test[i] == n:
            print(i, end = ' ')
    print()

#Считаем точность по всем данным
k=0
for i in range(10000):
    if np.argmax(predictions[i])==y_test[i]:
        k+=1
print('Accuracy:', k/10000)

#Считаем точность по пятеркам
def Accuracy_of_Num(N):
    m=0
    d=0
    l=0
    for i in range(10000):
        if y_test[i]==N:
            d+=1
            if np.argmax(predictions[i])==y_test[i]:
                m+=1
        elif np.argmax(predictions[i])==N:
            l+=1            
    print('m: Num of right defined N: ', m)
    print('d: Num of N:', d)
    print('l: Num of not 5 recognized as N:', l)
    print('m/d =', m/d)


#Визуальное представление тестов  
def Visualize(n):
    plt.figure()
    plt.imshow(x_test[n])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    print('I think it is...', np.argmax(predictions[n]))
    print('Right answer:', y_test[n])
