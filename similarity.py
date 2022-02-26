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
    # u: (active user id, its hidden factors)
    # UE: UΣ
    # k: top k (default=10)
    neighbor_list = []

    # 遍历计算每个用户和目标用户的相似度
    for i in range(len(UE)):
        if i == u[0]:
            # 跳过和自己计算
            continue
        similarity = cosine_similarity(u[1].reshape(
            1, -1), UE[i].reshape(1, -1))  # 计算相似度
        np.append(neighbor_list, (i, similarity))  # 添加计算结果到相似度列表

    neighbor_list.sort(key=lambda x: x[1])  # 升序排列相似度列表

    return neighbor_list[0:k]  # 返回top n列表
