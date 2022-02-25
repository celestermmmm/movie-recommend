'''
description: get neighbous similar with active user.

process: 1. compute similarities between active user and other users.
         2. choose top k in other users to be neighbous of active user.

date: 2022-01-30 15:47

author: Hanjiaxing
'''

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def select_neighbor(u, UE, k=10):
    # get top n nearist neighbors of u
    # u: active user
    # UE: UΣ
    # k: top k (default=10)
    neighbor_list = []
    for i in range(UE):
        similarity = cosine_similarity(u, UE[i])  # 计算相似度
        np.append(neighbor_list, (i, similarity))  # 添加计算结果到相似度列表
    np.delete(neighbor_list, (i, 0).index(
        neighbor_list), 0).sort()  # 删除u和自身计算的相似度(0)，升序排列相似度列表

    return neighbor_list[0:k]  # 返回top n列表
