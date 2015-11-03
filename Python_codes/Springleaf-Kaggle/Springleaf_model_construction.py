# coding: utf-8

#Python script for model construction via xgboost
#import Springleaf_data_preprocessing

import xgboost as xgb
import numpy as np
import pandas as pd
from numpy import nan as NA
from sklearn.cross_validation import train_test_split

nline_trn = 145232-1
nline_tst = 145233-1
nBatch = 20000
Y = pd.read_csv('./data/train_label.csv')
y_train = Y.values # read training labels
print('Loading train and test data from csv file...')
Xtrain = pd.read_csv("./data/train_cleaned.csv")
Xtest = pd.read_csv("./data/test_cleaned.csv")

ID = Xtest['ID']
Xtest.drop(['ID'], axis = 1, inplace = True)
print('Splitting data ...')
Xtrain_c, Xeval, Ytrain_c, Yeval = train_test_split(Xtrain.values, y_train, test_size = 0.3)

print('Loading train and test data into sparse matrix...')
# load it into xgboost sparse matrix
dtrain = xgb.DMatrix(data = Xtrain_c, missing = NA, label = Ytrain_c)
deval = xgb.DMatrix(data = Xeval, missing = NA, label = Yeval)
dtest = xgb.DMatrix(data = Xtest.values, missing= NA)  

print('Specifying parameters ...')
# specify parameters via map, definition are same as c++ version
param = {'max_depth': 11,
          'eta': 0.02, 
          'subsample': 0.7,
          'colsample_bytree': 0.8,
          'silent': 0, 
          'eval_metric': 'auc',
          'alpha': 0.0005,
          'lambda' : 1,
          'objective':'binary:logistic' }

# specify validations set to watch performance
watchlist  = [(deval,'eval'), (dtrain,'train')]
num_round = 950 #0.790154  #646 0.789907
print('Training ..')
bst = xgb.train(param, dtrain, num_round, watchlist)

# this is prediction
print('Prediction ...')
preds = bst.predict(dtest)

# save model
bst.save_model('./model/xgb.model')
bst.dump_model('./model/xgb_raw.txt')
 # save dmatrix into binary buffer
dtest.save_binary('./model/dtest.buffer')
dtrain.save_binary('./model/dtrain.buffer')
deval.save_binary('./model/deval.buffer')

submission = pd.DataFrame(data = preds, index = ID, columns = ['target'] )
print("Save to submission file")
submission.to_csv('./data/submission.csv')
