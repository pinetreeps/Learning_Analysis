# _*_ coding:utf-8 _*_
# Filename: .py
# Author: pang song
# python 3.6
# Date: 2019/02/01

'''
炮王的小任务，通过分数和各项评分，分析权重
'''

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    # excel_file = '/Users/pinetree_mac/ps_use/start_up_business/python数据分析/指标调查表/房产交易登记指标调查情况.csv'
    excel_file = '/Users/pinetree_mac/ps_use/start_up_business/python数据分析/指标调查表/开办企业指标调查情况(2).csv'
    house_data = pd.read_csv(excel_file)

    # 查看列
    print(house_data.columns.values)
    house_data = house_data.iloc[:, 2:6]
    print(house_data)
    # exit()

    # house_data.rename(columns={' 申请材料': '申请材料'}, inplace=True)

    print(house_data.corr())
    # exit()

    X_train, X_test, Y_train, Y_test = train_test_split(house_data.ix[:, 1:], house_data['分值'], train_size=.80)

    print("原始数据特征:", house_data.ix[:, 1:])

    print("原始数据特征:", house_data.ix[:, 1:].shape,
          ",训练数据特征:", X_train.shape,
          ",测试数据特征:", X_test.shape)

    print("原始数据标签:", house_data['分值'].shape,
          ",训练数据标签:", Y_train.shape,
          ",测试数据标签:", Y_test.shape )

    model = LinearRegression()

    model.fit(X_train, Y_train)

    a = model.intercept_  # 截距

    b = model.coef_  # 回归系数

    print("最佳拟合线:截距", a, ",回归系数：", b)

    score = model.score(X_test, Y_test)

    print(score)

    # 对线性回归进行预测

    Y_pred = model.predict(X_test)

    # print(Y_pred)
    
    # 绘制图形 预测值为蓝色实线，真实值为红色虚线
    plt.plot(range(len(Y_pred)), Y_pred, 'b', label="predict")
    plt.plot(range(len(Y_pred)), Y_test, 'r--', label="test")
    plt.legend()
    plt.show()


    exit()

    house_data.plot(kind="scatter", x="申请材料", y="分值")
    plt.show()
