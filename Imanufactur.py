##################################
#### Intelligent manufacturing ###
#### Author: CZB               ###
#### Date: 2017-12-22          ###
##################################

import numpy as np
import pandas as pd
import xlrd
from tqdm import tqdm
import os
from datetime import datetime
from time import mktime

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.svm import SVR
from sklearn.tree import ExtraTreeRegressor


def large_scale(df, value):
    for col in tqdm(df.columns):
        if df[col].sum() > value:
            df[col].apply(lambda x : x / 100000, inplace=True)
    return df

def large_col(df, value):
    large_col = []
    for col in tqdm(df.columns):
        if df[col].sum() > value:
            large_col.append(col)
    return large_col

#### calculate miss values
def col_miss(train_df):
    col_missing_df = train_df.isnull().sum(axis=0).reset_index()
    col_missing_df.columns = ['col', 'missing_count']
    col_missing_df = col_missing_df.sort_values(by='missing_count')
    return col_missing_df


#### obtain cols of XX type
def obtain_x(train_df, xtype):
    dtype_df = train_df.dtypes.reset_index()
    dtype_df.columns = ['col', 'type']
    return dtype_df[dtype_df.type == xtype].col.values


def totime(x):
    if x is not np.nan:
        if x > 1e15:
            time = datetime.strptime(str(x)[:-4], "%Y%m%d%H%M%S")
            return (time.hour * 3600 + time.minute * 60 + time.second)
        elif x > 1e13:
            time = datetime.strptime(str(x)[:-2], "%Y%m%d%H%M%S")
            # return (time.hour * 3600 + time.minute * 60 + time.second)
            return time


def date_cols(df):
    date_col = []
    for col in tqdm(df.columns):
        min = df[col].min()
        if min > 2010010100000000 and min < 2020010100000000:
            min = min / 100000000
        if min > 20200000:
            min = min / 1000000
        if min > 20000000 and min < 20200000:
            date_col.append(col)
    return date_col

def date_cols(df):
    date_col = []
    for col in tqdm(df.columns):
        min = df[col].min()
        if min > 2000000000000000:
            min = min / 100000000
        if min > 20200000:
            min = min / 1000000
        if min > 20000000 and min < 20200000:
            date_col.append(col)
    return date_col


def get_uniq(df, min, max):
    uniq_col = []
    for col in tqdm(df.columns):
        uniq = len(df[col].dropna().unique())
        if uniq >= min and uniq <= max:
            uniq_col.append(col)
    return uniq_col


def cal_corrcoef(float_df, y_train, float_col):
    corr_values = []
    for col in tqdm(float_col):
        corr_values.append(abs(np.corrcoef(float_df[col].values, y_train)[0, 1]))
    corr_df = pd.DataFrame({'col': float_col, 'corr_value': corr_values})
    corr_df = corr_df.sort_values(by='corr_value', ascending=False)
    return corr_df


def build_model(x, y):
    # reg_model = LinearRegression()
    # reg_model.fit(x_train,y_train)

    reg_model = RandomForestRegressor(n_estimators=500,
                                      oob_score=True,
                                      max_features="auto",
                                      min_samples_leaf=50)

    # reg_model = MLPRegressor(activation='relu',
    #                          hidden_layer_sizes=(50,30,20,20,20),
    #                          max_iter=700,
    #                          alpha=1e-05,
    #                          batch_size='auto',
    #                          beta_1=0.9,
    #                          beta_2=0.999,
    #                          early_stopping=False,
    #                          epsilon=1e-08,
    #                          learning_rate='constant',
    #                          learning_rate_init=0.001,
    #                          momentum=0.9,
    #                          nesterovs_momentum=True,
    #                          power_t=0.5,
    #                          random_state=1,
    #                          shuffle=True,
    #                          solver='lbfgs',
    #                          tol=0.0001,
    #                          validation_fraction=0.1,
    #                          verbose=False,
    #                          warm_start=False
    #                          )

    # reg_model = GradientBoostingRegressor(n_estimators=500)
    # reg_model = BaggingRegressor(n_estimators=30,
    #                              base_estimator=RandomForestRegressor(n_estimators=30))
    # reg_model = AdaBoostRegressor(n_estimators=30,
    #                              base_estimator=RandomForestRegressor(n_estimators=30))
    # reg_model = ExtraTreeRegressor()
    # reg_model = SVR()
    # reg_model = AdaBoostRegressor(n_estimators=500)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    mse = cross_val_score(reg_model,
                          x_train,
                          y_train,
                          cv=12,
                          scoring='neg_mean_squared_error',
                          n_jobs=8)
    print(mse)
    print(mse.mean())
    reg_model.fit(x, y)
    return reg_model


if __name__ == '__main__':
    if os.path.exists('data_train.csv') is True \
            and os.path.exists('data_testa.csv') is True:
        data_train = pd.read_csv('data_train.csv')
        x_train = data_train.iloc[:, :-1].values
        y_train = data_train.Y.values
        x_test = pd.read_csv('data_testa.csv').values
    elif os.path.exists('x_train.npy') is True \
            and os.path.exists('y_train.npy') is True \
            and os.path.exists('x_test.npy') is True:
        x_train = np.load('x_train.npy')
        y_train = np.load('y_train.npy')
        x_test = np.load('x_test.npy')
    else:
        # read train data
        print('read train...')
        train_df = pd.read_excel('train.xlsx')
        print('read test...')
        test_df = pd.read_excel('submit_A.xlsx')
        print('train shape:', train_df.shape)

        # 训练集结果
        y_train = train_df.Y.values

        # 训练集数据拼接并删除ID列
        train_concat = pd.concat([train_df.drop(['Y'], axis=1), test_df])\
                         .drop(['ID'], axis=1)

        # del cols that miss number greater than 299
        col_missing_df = col_miss(train_concat)
        almost_missing_columns = col_missing_df[col_missing_df.missing_count >= 299].col.values
        print('number of almost miss col:', len(almost_missing_columns))
        train_concat.drop(almost_missing_columns, axis=1, inplace=True)
        print('deleted,and train_concat shape:', train_concat.shape)

        # 删除重复列
        # temp = train_concat.select_dtypes(include=['int64', 'float64'])
        # train_concat.drop(get_uniq(temp, 1), axis=1, inplace=True)
        train_concat.drop(get_uniq(train_concat, 1, 1), axis=1, inplace=True)

        # 收集离散列
        temp = train_concat.select_dtypes(exclude=['int64', 'float64']).columns
        # TODO: 暂且转化成ID值
        train_concat[temp] = train_concat[temp].apply(lambda x: pd.factorize(x)[0])
        # train_concat.drop(temp.columns, axis=1, inplace=True)
        # train_df.filter(float64_date_col).apply(lambda x: x.apply(lambda y: totime(y)))

        # 收集日期列
        temp = train_concat.select_dtypes(include=['int64', 'float64'])
        # TODO: 暂且删除处理
        train_concat.drop(date_cols(temp), axis=1, inplace=True)

        # fill nan
        # print('get float cols data and fill nan...')
        print('filled nan')
        train_concat.apply(lambda x: x.fillna(x.median(), inplace=True))

        train_concat.apply(lambda x: pd.to_numeric(x, downcast='float'), inplace=True)

        # 处理过大数据
        # TODO: 暂且删除处理
        # train_concat.drop(large_col(train_concat, 20000000), axis=1, inplace=True)

        # obtained corrcoef greater than 0.2

        #        corr_df = cal_corrcoef(float_df,y_train,float64_col)
        #        corr02 = corr_df[corr_df.corr_value>=0.2]
        #        corr02_col = corr02['col'].values.tolist()
        #        x_train = float_df[corr02_col].values

        print('get x_train')
        x_train = train_concat[:500].values
        print('get test data...')
        x_test = train_concat[500:].values
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)
        X = np.vstack((x_train, x_test))
        # X = preprocessing.scale(X)
        x_train = X[0:len(x_train)]
        x_test = X[len(x_train):]
        np.save('x_train.npy', x_train)
        np.save('y_train.npy', y_train)
        np.save('x_test.npy', x_test)
    print('build model...')
    model = build_model(x_train, y_train)
    print('predict and submit...')
    subA = model.predict(x_test)
    # read submit data
    sub_df = pd.read_csv('subA.csv', header=None)
    sub_df['Y'] = subA
    directory = os.environ['HOME'] + '/tianchi/data/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    str_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    joblib.dump(model, directory + str_time + ".mod")
    sub_df.to_csv(directory + str_time + '-res.csv', header=None, index=False)
