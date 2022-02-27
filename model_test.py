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


def get_precision(test_data):

    Ratings = load_rating_data()  # 加载(用户,电影,评分)矩阵

    Ratings_origin = copy.deepcopy(Ratings)  # 拷原始用户-电影评分矩阵用以保留

    svd_computer = SVD(ratings=Ratings)
    svd_computer.train()  # 对U矩阵进行svd, 默认取前100%奇异值
    p = svd_computer.user_mat

    # 计算电影总数
    item_list = [i[1] for i in Ratings]
    movie_count = 0
    for i in item_list:
        movie_count += 1/item_list.count(i)

    U_predict = rating_predict(
        Ratings_origin, p, int(movie_count))  # 预测评分(四舍五入取整)

    # 测试准确率
    correct = 0
    for i in U_predict:
        if i in test_data:
            correct += 1
    precision = correct/len(test_data)

    return precision
    # recommend_n_movie(U_predict, len(p))  # 打印TopN推荐电影结果
