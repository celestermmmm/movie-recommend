'''
description: input and init data.

process: 1. open csv files
         2. read line
         3. handle data format initially
         4. store data into a list
         5. return list       

date: 2022-01-30 15:30

author:zhaorong
'''

import numpy as np
from sklearn import model_selection
import random
import gc


def load_rating_data(file_path):
    """
    load movie lens 100k ratings from original rating file.

    """
    prefer_matrix = []
    # user_num = 1  # 用户游标
    # movie_num = 0  # 电影游标
    # user_record = []  # 一个用户的所有评分记录
    # rec = []
    f = open(file_path, 'r')
    lines = f.readlines()
    del lines[0]  # 去掉表头

    for i in range(len(lines)):  # 遍历原始表格

        (_, userid, movieid, rating, ts) = lines[i].split(',')  # 按行读取其中4项元素: 用户id, 电影id, 电影评分, 时间戳
        uid = int(userid)
        mid = int(movieid)
        rat = float(rating)

        prefer_matrix.append([uid, mid, rat])

        # """
        # 比对用户游标, 将一个用户的所有评分信息读取完毕之后, 移动用户游标
        # """
        # if uid == user_num:  # 未收集完当前用户信息且不是最后一条信息
        #     user_record.extend([0 for i in range(mid-movie_num-1)])
        #     user_record.append(rat+1)
        #     movie_num = mid
        # elif uid > user_num:  # 上一用户信息收集完，user_record尾部补零
        #     user_record.extend([0 for j in range(205106-movie_num)])

        #     # rec.append(len(user_record))
        #     # if len(user_record)!=205106:
        #     #     print(i+1)
        #     #     print(rec)
        #     #     raise Exception("列表长度错误")
        #     # 205106为电影的最大编号

        #     prefer_matrix.append(user_record)
        #     user_num += 1

        #     # 以上完成了一组输入
        #     user_record = []
        #     user_record.extend([0 for j in range(mid-1)])
        #     user_record.append(rat+1)
        #     movie_num = mid
        #     gc.collect()
        # else:  # 该条信息是最后一个用户的最后一条信息
        #     user_record.extend([0 for j in range(mid-movie_num-1)])
        #     user_record.append(rat+1)
        #     user_record.extend([0 for j in range(205106-mid)])
        #     prefer_matrix.append(user_record)

    data = np.array(prefer_matrix)
    return data
