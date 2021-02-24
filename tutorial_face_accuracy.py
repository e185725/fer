
#現状の知識のみで感情分類を行ってみる

###ライブラリなどの準備
import sys
sys.path.append('..')
import read_data
import glob 
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd 

#なんかエラー出た
"""
I tensorflow/core/platform/cpu_feature_guard.cc:142] 
This TensorFlow binary is optimized with oneAPI Deep Neural Network Library 
(oneDNN) to use the following CPU instructions in performance-critical operations:  
AVX2 FMA
"""
#対処法
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#テストデータと訓練データを格納するための配列を用意する
train_data,train_label = read_data.read_data( read_data.train_name )
test_data ,test_label = read_data.read_data( read_data.test_file )

#画素値を0~1に変更している。（仕様上)
train_data,test_data = train_data/255.0,test_data/255.0

#画像の可視化

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_data[-i], "gray")
#     plt.xlabel(train_label[i])
# plt.show()



###モデルの作成
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(48, 48)),
  #入力画像48×48の二次元配列を一次元配列に変換
  tf.keras.layers.Dense(128, activation='relu'),
  #隠れ層128個のノード
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(7)
  #出力層に当たる
])
"""
reluとは
　AI／機械学習のニューラルネットワークにおける
ReLU（Rectified Linear Unit、「レルー」と読む）とは、
関数への入力値が0以下の場合には出力値が常に0、
入力値が0より上の場合には出力値が入力値と同じ値となる関数である。
f_relu(x) = x^+ = max(0,x)
"""

"""
dropoutとは
更新の時のノードのうちのいくつかを無効にして学習を行うことによって、
過学習を避ける役割がある。
"""

"""
>>>model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 2304)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               295040    
_________________________________________________________________
dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 7)                 903       
=================================================================
Total params: 295,943
Trainable params: 295,943
Non-trainable params: 0
_________________________________________________________________
None
"""

"""
%   brew install graphviz
%   pip3 install pydot 
ターミナル上で実行してインストールする必要がある
"""
#モデルを可視化
# plot_model(
#     model,
#     show_shapes=True,
#     show_layer_names=True,
#     to_file="model.png"
# )
# img = cv2.imread("model.png")
# cv2.imshow("img",img)
# cv2.waitKey(0)

"""
#実行テスト
predictions = model(train_data[:1]).numpy()
print(predictions)
print(tf.nn.softmax(predictions).numpy())
"""

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
"""
SparseCategoricalCrossentropyは、ワンホットではなく整数ラベルをとる.
[0,1,2] => o
[1,0,0],[0,1,0],[0,0,1] => x
"""

#モデルをコンパイル
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
"""
adamとは、モーメンタムとRMSPropを合わせたもの
ここはもう少し調べる必要がある
"""

#モデルで学習
epochs = 10
result = model.fit(train_data, train_label, epochs=epochs,validation_data=(test_data,test_label),shuffle=True)

"""
>>>print(result.history.keys())
dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
"""

#正解率の可視化
# plt.plot(range(1, epochs+1), result.history['accuracy'], label="training")
# plt.plot(range(1, epochs+1), result.history['val_accuracy'], label="validation")
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

#入力データに対する出力結果を確率に変換することで各クラスの確率を表現している。
model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

"""
>>>print(probability_model(test_data[:5]))
tf.Tensor(
[[0.11418401 0.00580847 0.1347376  0.14201069 0.19748609 0.08860935
  0.3171637 ]
 [0.14225979 0.02284919 0.15572166 0.15900636 0.18834594 0.09493577
  0.23688132]
 [0.15582033 0.01279152 0.14378975 0.38517007 0.11481778 0.08002264
  0.10758793]
 [0.09194526 0.00521894 0.06582114 0.70756924 0.0458149  0.02881109
  0.05481929]
 [0.18003854 0.01328593 0.24466953 0.1195146  0.14057928 0.23389447
  0.06801767]], shape=(5, 7), dtype=float32)

"""

###予測値を実際に見てみた
predictions = model.predict(test_data)
# for i in range(5):
#     #argmaxで二次元配列の列ごとの最大値を示すインデックスを返す
#     #予測した値と実際の解
#     print(np.argmax(predictions[i]),test_label[i])


###ヒートマップの表示と保存
emotion = ["angry","disgust","fear","happy","neutral","sad","surprise"]
pred = [np.argmax(i) for i in predictions]
cm = confusion_matrix(test_label, pred)
#print(len(test_label))
#print(cm)
cm = pd.DataFrame(data=cm, index=emotion, 
                           columns= emotion)
sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues',fmt="g")
plt.xlabel("Pre", fontsize=13)
plt.ylabel("True", fontsize=13)
#plt.show()
plt.savefig('sklearn_confusion_matrix.png')

###正答率のグラフ化
predictions = model.predict(test_data[:5])
for i in range(5):
    #argmaxで二次元配列の列ごとの最大値を示すインデックスを返す
    #予測した値と実際の解
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    bar_label = [0,1,2,3,4,5,6]
    axs[0].imshow(test_data[i],"gray")
    axs[0].set_title(i)
    axs[1].bar(bar_label,predictions[i],color="orange",alpha = 0.7)
    axs[1].grid()
    #print(predictions[i],test_label[i])
plt.show()









