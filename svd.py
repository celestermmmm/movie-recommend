'''
description: singular value decomposition;
             culculate the matrix P and Q which product is  approximately equal to the initial user-movie matrix .

process: 1. initialize P and Q
         2. iterate P and Q according to the train model
         3. generate P and Q

date: 2022-01-30 15:38

author: Hanjiaxing
'''

import numpy as np


# def svd_matrix(data, x=1):
#     '''
#     data: the user-movie matrix
#     x: top % singular value, default = 100%
#     '''
#     u, sigma, _ = np.linalg.svd(data)  # 计算svd
#     sigma = np.diag(sigma)  # np.svd直接输出一维向量Σ, 需手动变成对角矩阵
#     singular_num = int(np.shape(u)[0]*x)  # 计算选取的奇异值数目
#     u_reduced = u[:, :singular_num]  # 降维U矩阵
#     sigma_reduced = sigma[:singular_num, :singular_num]  # 降维Σ矩阵

#     return np.dot(u_reduced, sigma_reduced)  # 返回降维后的UΣ矩阵


class SVD:

    """
    ratings: 训练数据, n*3数组 (user, item, rating)
    K: 隐因子维数
    Lambda: 惩罚系数
    gamma: 学习率
    steps: 迭代次数
    """

    def __init__(self, ratings, logger=None,  K=40, Lambda=0.005, gamma=0.02, steps=40):
        # 字符串数组转换成数字数组
        # user, item字符串映射为数字
        self.ratings = []
        self.user2id = {}
        self.item2id = {}
        # user已推荐过的items
        self.userRecItems = {}
        user_id = 0  # 用户游标
        item_id = 0  # 电影游标


        for user, item, r in ratings:
            if user not in self.userRecItems:
                self.userRecItems[user] = set()

            new_tup = []
            if user not in self.user2id:
                self.user2id[user] = user_id
                new_tup.append(user_id)
                user_id += 1
            else:
                new_tup.append(self.user2id[user])

            if item not in self.item2id:
                self.item2id[item] = item_id
                self.userRecItems[user].add(item_id)
                new_tup.append(item_id)
                item_id += 1
            else:
                self.userRecItems[user].add(self.item2id[item])
                new_tup.append(self.item2id[item])

            new_tup.append(r)
            self.ratings.append(new_tup)

        self.ratings = np.array(self.ratings)

        user_num = len(self.user2id.keys())
        item_num = len(self.item2id.keys())

        self.user_mat = 0.1 * np.random.randn(user_num, K) / np.sqrt(K)
        self.item_mat = 0.1 * np.random.randn(K, item_num) / np.sqrt(K)
        self.bias_user = np.array([0.0] * user_num)
        self.bias_item = np.array([0.0] * item_num)
        self.global_mean = np.mean(ratings[:, 2])
        self.Lambda = Lambda
        self.gamma = gamma
        self.steps = steps
        self.logger = logger

    def train(self):
        losses = []
        for step in range(self.steps):
            loss = 0
            np.random.shuffle(self.ratings)

            for i, j, r in self.ratings:
                i = int(i)
                j = int(j)
                Err = r - (np.dot(self.user_mat[i, :], self.item_mat[:, j]) +
                           self.global_mean + self.bias_user[i] + self.bias_item[j])
                loss += Err**2

                self.user_mat[i, :] += self.gamma * \
                    (Err * self.item_mat[:, j] -
                     self.Lambda * self.user_mat[i, :])
                self.item_mat[:, j] += self.gamma * \
                    (Err * self.user_mat[i, :] -
                     self.Lambda * self.item_mat[:, j])
                self.bias_user[i] += self.gamma * \
                    (Err - self.Lambda * self.bias_user[i])
                self.bias_item[j] += self.gamma * \
                    (Err - self.Lambda * self.bias_item[j])

            self.gamma *= 0.9

            loss += self.Lambda * ((self.user_mat*self.user_mat).sum() + (self.item_mat*self.item_mat).sum()
                                   + (self.bias_user*self.bias_user).sum() + (self.bias_item*self.bias_item).sum())
            losses.append(loss)
            #print('step '+str(step)+' loss '+str(loss))
            if self.logger is not None:
                self.logger.info('step '+str(step)+' loss: '+str(loss))

            if self.isConverged(losses):
                break

    def normalize(self, ratings):
        mean = np.mean(ratings)
        std = np.std(ratings)
        new_ratings = []
        for r in ratings:
            new_ratings.append((r-mean)/std)

        return np.array(new_ratings)

    def isConverged(self, losses, last_n_steps=30):
        last_losses = losses[-last_n_steps:]
        is_descending = False
        if len(last_losses) >= last_n_steps and np.std(last_losses) < 0.001:
            is_descending = True
        return is_descending

    def early_stopping(self, losses):
        # use early stopping
        early_stopping_rounds = int(self.steps / 10)
        is_descending = True
        last_losses = losses[-early_stopping_rounds:]

        if len(last_losses) >= early_stopping_rounds:
            for i in range(len(last_losses)-1):
                if losses[i+1] - losses[i] > 0.001:
                    is_descending = False
                    break
        else:
            is_descending = False
        return is_descending
