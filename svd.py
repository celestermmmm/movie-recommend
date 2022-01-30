'''
description: singular value decomposition.

process: 1. decomposite into UΣ
         2. choose front x% feature value
         3. generate reduced matrix

date: 2022-01-30 15:38

author: Hanjiaxing
'''

import numpy as np


def svd_matrix(data, x=1):
    '''
    data: the user-movie matrix
    x: top % singular value, default = 100%
    '''
    u, sigma, _ = np.linalg.svd(data)  # 计算svd
    singular_num = int(np.shape(u)[0]*x)  # 计算选取的奇异值数目
    u_reduced = u[:, 0:singular_num]  # 降维U矩阵
    sigma_reduced = sigma[0:singular_num, 0:singular_num]  # 降维Σ矩阵

    return np.dot(u_reduced, sigma_reduced)  # 返回降维后的UΣ矩阵
