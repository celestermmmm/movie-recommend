from sklearn.model_selection import train_test_split
import pandas as pd

# 计算不同用户个数


def count_num(s):
    li = list(set(s))
    num = len(li)
    return num


k = 1
while k:
    file_path = '../movie-recommend-1/rtest_0.csv'
    data = pd.read_csv(file_path, sep=',', index_col=None)
    # 随机切割数据集
    ntrain, ntest = train_test_split(data, test_size=0.1)

    # 记录数据集中所有用户
    train_user_list = [i[1] for i in ntrain.itertuples()]
    test_user_list = [i[1] for i in ntest.itertuples()]

    # 计算不同用户个数
    num_train = count_num(train_user_list)
    num_test = count_num(test_user_list)

    # 用户个数相等时输出,否则再次划分
    if num_test == num_train:
        k = k-1

# 存储数据集结果
ntrain.to_csv('../movie-recommend-1/ntrain.csv')
ntest.to_csv('../movie-recommend-1/ntest.csv')
