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
from sklearn.neural_network import MLPRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib


#### calculate miss values
def col_miss(train_df):
    col_missing_df = train_df.isnull().sum(axis=0).reset_index()
    col_missing_df.columns = ['col','missing_count']
    col_missing_df = col_missing_df.sort_values(by='missing_count')
    return col_missing_df

#### obtain cols of XX type
def obtain_x(train_df,xtype):
    dtype_df = train_df.dtypes.reset_index()
    dtype_df.columns = ['col','type']
    return dtype_df[dtype_df.type==xtype].col.values

def totime(x):
  if x is not np.nan:
    if x > 1e15:
      time = datetime.strptime(str(x)[:-4], "%Y%m%d%H%M%S")
      return (time.hour * 3600 + time.minute * 60 + time.second)
    elif x > 1e13:
      time = datetime.strptime(str(x)[:-2], "%Y%m%d%H%M%S")
      return (time.hour * 3600 + time.minute * 60 + time.second)


def date_cols(train_df,float_col):
    float_date_col = []
    for col in float_col:
        if train_df[col].min() > 1e13:
#            float_date_col.append(col)
#    return float_date_col
            train_df[col] = train_df[col].map(lambda x:totime(x))
    return float_col

def float_uniq(float_df,float_col):
    float_uniq_col = []
    for col in tqdm(float_col):
        uniq = float_df[col].unique()
        if len(uniq) == 1:
            float_uniq_col.append(col)
    return float_uniq_col

def cal_corrcoef(float_df,y_train,float_col):
    corr_values = []
    for col in float_col:
        corr_values.append(abs(np.corrcoef(float_df[col].values,y_train)\
                [0,1]))
    corr_df = pd.DataFrame({'col':float_col,'corr_value':corr_values})
    corr_df = corr_df.sort_values(by='corr_value',ascending=False)
    return corr_df

def build_model(x,y):
    # reg_model = LinearRegression()
    # reg_model.fit(x_train,y_train)
    # reg_model = RandomForestRegressor(n_estimators=1000,
    #                                   max_features=None)
    # reg_model = MLPRegressor(activation='relu',
    #                          hidden_layer_sizes=(120,100),
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

    reg_model = GradientBoostingRegressor(random_state=10)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    mse = cross_val_score(reg_model,
                          x_train,
                          y_train,
                          cv=20,
                          scoring='neg_mean_squared_error',
                          n_jobs=8)
    print(mse)
    print(mse.mean())
    reg_model.fit(x,y)
    return reg_model

if __name__ == '__main__':
    if os.path.exists('x_train.npy') is True \
        and os.path.exists('y_train.npy') is True\
        and os.path.exists('x_test.npy') is True:
        x_train = np.load('x_train.npy')
        y_train = np.load('y_train.npy')
        x_test = np.load('x_test.npy')
    else:
        # read train data
        print('read train...')
        train_df = pd.read_excel('train.xlsx')
        print('train shape:',train_df.shape)
        # calculate the number of miss values
        col_missing_df = col_miss(train_df)
        # del cols of all nan
        all_nan_columns = col_missing_df[col_missing_df.missing_count==499].\
                            col.values
        print('number of all nan col:',len(all_nan_columns))
        train_df.drop(all_nan_columns,axis=1,inplace=True)
        print('deleted,and train shape:', train_df.shape)
        # obtain float cols
        float64_col = obtain_x(train_df,'float64')
        print('obtained float cols, and count:',len(float64_col))
#        # del cols that miss number greater than 200
#        miss_float = train_df[float64_col].isnull().sum(axis=0).reset_index()
#        miss_float.columns = ['col','count']
#        miss_float_almost = miss_float[miss_float['count']>200].col.values
#        float64_col = float64_col.tolist()
#        float64_col = [col for col in float64_col if col not in \
#                        miss_float_almost]
#        print('deleted cols that miss number > 200')
        # del date cols
        float64_date_col = date_cols(train_df,float64_col)
#        float64_col = [col for col in float64_col if col not in\
#                        float64_date_col]
#        print('deleted date cols, and number of float cols:',len(float64_col))
        # fill nan
        print('get float cols data and fill nan...')
        float_df = train_df[float64_col]
        float_df.fillna(float_df.median(),inplace=True)
        print('filled nan')
        # del cols which unique eq. 1
        float64_uniq_col = float_uniq(float_df,float64_col)
        float64_col = [col for col in float64_col if col not in\
                        float64_uniq_col]
        print('deleted unique cols, and float cols count:',len(float64_col))
        # obtained corrcoef greater than 0.2
        float64_col.remove('Y')
        y_train = train_df.Y.values

#        corr_df = cal_corrcoef(float_df,y_train,float64_col)
#        corr02 = corr_df[corr_df.corr_value>=0.2]
#        corr02_col = corr02['col'].values.tolist()
        print('get x_train')
#        x_train = float_df[corr02_col].values
        x_train = float_df[float64_col].values
        print('get test data...')
        test_df = pd.read_excel('submit_A.xlsx')
        sub_test = test_df[float64_col]
        sub_test.fillna(sub_test.median(),inplace=True)
        x_test = sub_test.values
        print('x_train shape:',x_train.shape)
        print('x_test shape:',x_test.shape)
        print('build model...')
        X = np.vstack((x_train,x_test))
        X = preprocessing.scale(X)
        x_train = X[0:len(x_train)]
        x_test = X[len(x_train):]
        np.save('x_train.npy', x_train)
        np.save('y_train.npy', y_train)
        np.save('x_test.npy', x_test)
    model = build_model(x_train, y_train)
    print('predict and submit...')
    subA = model.predict(x_test)
    # read submit data
    sub_df = pd.read_csv('subA.csv',header=None)
    sub_df['Y'] = subA
    path = os.environ['HOME'] + '/tianchi/data/'
    str_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    joblib.dump(model, path + str_time + ".mod")
    sub_df.to_csv(path + str_time + '-res.csv',header=None,index=False)
