from data_input import load_rating_data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from itertools import count



k=1
while k:
    file_path='../movie_recommend/data_train.csv'
    data=pd.read_csv(file_path)
    ntrain,ntest=train_test_split(data,test_size=0.1)
    ntr = load_rating_data(file_path='../movie_recommend/ntrain.csv')
    nte = load_rating_data(file_path='../movie_recommend/ntest.csv')

    train_user_list = [i[0] for i in ntr]
    test_user_list=[i[0] for i in nte]

    def count_num(list):
        li=list(set(list))
        num=count(li)
        return num

    num_train=count_num(train_user_list)
    num_test=count_num(test_user_list)

    if num_test==num_train:
        k=k-1

ntrain.to_csv('../movie_recommend/ntrain.csv')
ntest.to_csv('../movie_recommend/ntest.csv')