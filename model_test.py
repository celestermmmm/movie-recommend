'''
description: test model.

process: 

date: 2022-01-30 16:03

author: Hanjiaxing
'''

from data_input import load_rating_data
from svd import SVD
from recommend import rating_predict
import copy
import numpy as np
from sklearn.metrics import mean_squared_error


def get_movie_set(file_path='../rtest_0.csv'):
    prefer_matrix = []
    f = open(file_path, 'r')
    lines = f.readlines()
    del lines[0]  # 去掉表头

    for i in range(len(lines)):  # 遍历原始表格
        (userid, movieid, rating, ts) = lines[i].split(
            ',')  # 按行读取其中4项元素: 用户id, 电影id, 电影评分, 时间戳
        uid = int(userid)
        mid = int(movieid)
        rat = float(rating)

        prefer_matrix.append([uid, mid, rat])

    movie_set = set([i[1] for i in prefer_matrix])

    return movie_set


def get_rmse(test_data):
    '''
    获取RMSE
    '''
    Ratings = load_rating_data(
        file_path='../movie-recommend/ntrain.csv')  # 加载(用户,电影,评分)矩阵

    Ratings_origin = copy.deepcopy(Ratings)  # 拷贝原始用户-电影评分矩阵用以保留
    test_data_list = copy.deepcopy(test_data).tolist()  # 拷贝测试集用来转化成list

    svd_computer = SVD(ratings=Ratings)
    svd_computer.train()  # 对U矩阵进行svd
    p = svd_computer.user_mat
    q = svd_computer.item_mat

    pq = np.dot(p, q)

    noncolla_item = []
    for u_id in range(len(pq)):
        for m_id in range(len(pq[u_id])):
            noncolla_item.append([u_id+1, m_id+1, pq[u_id][m_id]])

    noncolla_pre_item = []
    for i in test_data:
        exist = 0
        for item in noncolla_item:
            if i[0] == item[0] and i[1] == item[1]:
                noncolla_pre_item.append(item)
                exist = 1
                break
        if exist == 0:
            test_data_list.remove(i.tolist())

    noncolla_pre_item = sorted(noncolla_pre_item, key=(lambda x: [x[0], x[1]]))
    uncolla_rmse = np.sqrt(mean_squared_error(
        noncolla_pre_item, test_data_list))

    movie_set = get_movie_set()  # 获取训练集和测试集中全部movie_id的集合
    U_predict = rating_predict(
        Ratings_origin, p, movie_set)  # 预测评分

    # 求RMSE
    predict_item = []
    for i in test_data:
        exist = 0
        for item in U_predict:
            if i[0] == item[0] and i[1] == item[1]:
                predict_item.append(item)
                exist = 1
                break

                # try:

                #     for item in U_predict:
                #         if item[0] == i[0]:
                #             predict_item = item

                #         else:
                #             print(item)
                #             return
                # continue
                #predict_item = [item for item in U_predict if item[0]==i[0] and item[1]==i[1]][0]
                # except IndexError as ie:
                #     print("发生时刻：", i)
                #     print("Indexerror", ie)
        if exist == 0:
            print("i: ", i)

    # 按照1.userid 2.itemid的顺序保持预测结果以及测试集同序
    predict_item = sorted(predict_item, key=(
        lambda x: [x[0], x[1]]))
    # noncolla_pre_item = sorted(noncolla_pre_item, key=(
    #     lambda x: [x[0], x[1]]))
    test_data = sorted(test_data, key=(
        lambda x: [x[0], x[1]]))

    colla_rmse = np.sqrt(mean_squared_error(predict_item, test_data))
    # uncolla_rmse = np.sqrt(mean_squared_error(noncolla_pre_item, test_data))

    return colla_rmse, uncolla_rmse


# recommend_n_movie(U_predict, len(p))  # 打印TopN推荐电影结果
path ='../movie-recommend/ntest.csv'
test_data = load_rating_data(path)

# 获取有协同过滤方法的rmse和无协同过滤方法的rmse
colla_rmse, uncolla_rmse = get_rmse(test_data)
print("Collaboration rmse: ", colla_rmse)
print("Uncolla rmse: ", uncolla_rmse)
