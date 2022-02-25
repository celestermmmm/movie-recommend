'''
description: test model.

process: 

date: 2022-01-30 16:03

author: Hanjiaxing
'''

from data_input import load_rating_data
from svd import svd_matrix
from recommend import rating_predict, recommend_n_movie
import copy


def main():

    U = load_rating_data()  # 加载用户-电影评分矩阵
    U_origin = copy.deepcopy(U)  # 拷原始用户-电影评分矩阵用以保留

    UE = svd_matrix(U)  # 对U矩阵进行svd, 默认取前100%奇异值

    U = rating_predict(U, UE)  # 预测评分并填充U矩阵

    recommend_n_movie(U_origin, U)  # 打印TopN推荐电影结果


if __name__ == "__main__":
    main()
