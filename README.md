# fer (Facial Expression Recognition )

## Overview 
画像から表情を機械学習を用いて予測するプログラム（正答率は約30%)
0="angry",1="disgust",2="fear",3="happy",4="sad",5="surprise",6="neutral"
## Requirement
- Python3

## Usage
Kaggle(FER-2013)
[text](https://datarepository.wolframcloud.com/resources/FER-2013)
のファイルをダウンロードして、csvファイルから画像ファイルに変換後、モデル構築、学習、予想させる。
画像変換はgene\_record.py、画像読み込みはread\_data.pyから行うことができる

## Features

#Model
![tutorial_face_model](https://user-images.githubusercontent.com/44591782/108961875-4e53aa80-76bb-11eb-848d-0dd5d3b8cf83.png)

#Acuraccy
![tutorial_face_accuracy](https://user-images.githubusercontent.com/44591782/108962807-a0490000-76bc-11eb-822d-5d2bf7caf51a.png)

#heat-map
![sklearn_confusion_matrix](https://user-images.githubusercontent.com/44591782/108963697-cae78880-76bd-11eb-8e25-fc450bd197b5.png)

#result
![Figure_6](https://user-images.githubusercontent.com/44591782/108964641-1189b280-76bf-11eb-8379-9113624cc667.png)

## Licence
MIT Licence
