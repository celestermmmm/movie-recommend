'''
description: generate recommendation list.

process: 1. predict scores of empty cell in user-movie matrix
         2. choose top n movies by predicted scores.

date: 2022-01-30 15:52

author: Hanjiaxing
'''

import numpy as np
import pandas as pd
from similarity import select_neighbor


def rating_predict(U, p, movie_set):
    # fill up user-movie matrix by predicted rating
    # U: (user, movie, rating) matrix
    # p: user-hidden_factor matrix
    # movie_set: set of movie id

    U_predict = []
    for uid in range(len(p)):

        u_marked = [i[1] for i in U if i[0] == uid+1]
        neighborList = select_neighbor((uid, p[uid]), p)

        Ru = np.mean(u_marked)  # 求该用户的平均值Ru

        for item_i in movie_set:
            if item_i not in u_marked:
                sim_rvi_rv_list = [([i[2] for i in U if i[0] == neighbor[0] and i[1] == item_i][0] - np.mean([i[2] for i in U if i[0] == neighbor[0]]))*neighbor[1]
                                   for neighbor in neighborList if [i[2] for i in U if i[0] == neighbor[0] and i[1] == item_i][0] != 0]
                if len(sim_rvi_rv_list) == 0:
                    # 如果所有的邻居都没有对这部电影评分，则对该电影的分数预测为0
                    U_predict.append([uid+1, item_i, 0])
                else:
                    # 计算Σsim(u,v)(rvi-rv), v in K
                    sum_sim_rvi_rv = sum(sim_rvi_rv_list)
                    # 计算Σ|sim(u,v)|, v in K
                    sum_abs_sim_u_v = sum(abs(neighbor[1])
                                          for neighbor in neighborList)

                    # 计算P(u,i)
                    P_ui = Ru+sum_sim_rvi_rv/sum_abs_sim_u_v
                    U_predict.append([uid+1, item_i, P_ui])
    return np.array(U_predict, dtype=float)


def recommend_n_movie(U, user_count, n=10):
    # recommend top n movies for active user
    # U: predicted user-movie matrix
    # user_count: account of users
    # n: top n movie (default=10)

    for i in range(user_count):
        rating_list = [(u[1], u[2]) for u in U if u[0] == i+1]

        rating_list.sort(key=lambda x: -x[1])
        if len(rating_list) < n:
            print([e[1] for e in rating_list])
        else:
            print([e[1] for e in rating_list][:n])

    return
