
#入力データの読み込み
#train_dataのデータ数はこの様になっている
# 3994
# 436
# 4097
# 7215
# 4830
# 3171
# 4965

import glob 
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.utils import plot_model

#訓練データ用の画像をtrain_dataにまとめるコード
###カレントディレクトリから行なっているので後で修正する
train_name = "../Training/{}/*"
test_file = "../PublicTest/{}/*"
test_file = "../PrivateTest/{}/*"

def read_data(file_name):
    """
    テストデータと訓練データを格納するための
    配列を用意する
    """
    data = []
    label = np.array([])

    #data 読み込み

    for i in range(7):  
        files = glob.glob(file_name.format(str(i)))
        print(len(files))
        for v,file in enumerate(files):
            ##　opencvのimreadを用いることでnp.arrayで画像データを表している
            img = cv2.imread(file,0)#第二引数は0でgrayscale化
            #print(v,file)

            """
            >>>img.shape
            (48, 48)
            """

            data.append(img)
            label =  np.append(label,i)

    data = np.array(data)

    return data,label

