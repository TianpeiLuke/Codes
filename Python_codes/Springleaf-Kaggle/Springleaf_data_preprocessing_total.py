# coding: utf-8

# Python script for Springleaf data preprocessing
# https://www.kaggle.com/c/springleaf-marketing-response
# modify to load the whole data
# This script load data in sequential manner

import numpy as np
from numpy import nan as NA
from pandas import Series, DataFrame
import pandas as pd
from sklearn import preprocessing

#import re
#import time

# Read training data

# In[138]:

nline_trn = 145232 - 1
nline_tst = 145233 - 1
nBatch = 20000
y_train = np.zeros([nline_trn, 1])
# read the outlier lists
all_nan  = pd.read_csv('./data/list_all_nan.csv')
outlier_num = pd.read_csv('./data/outlier_num.csv')
outlier_bool =  pd.read_csv('./data/outlier_bool.csv')
outlier_str = pd.read_csv('./data/outlier_str.csv')
outlier_frq_str = pd.read_csv('./data/outlier_freq_str.csv')
# read the list of time featurs
time_feature = pd.read_csv('./data/list_time.csv')
most_common_str= pd.read_csv('./data/list_common_str4.csv')

rlist = range(np.ceil(float(nline_trn)/float(nBatch)).astype(int)) 
for i in rlist:
#reading data in batches
    print "===================================================="
    print "run %i: Reading..." % i
    if i == 0:
        train = pd.read_csv("./data/train.csv", nrows=nBatch)
    else: 
        train = pd.read_csv("./data/train.csv", nrows=nBatch,
                            skiprows= range(1,i*nBatch+1))

    nrows = len(train.index)
    ncols  = len(train.columns)

    print('Row count: %d' % nrows)
    print('Column count: %d' % ncols)

    print("Row count in total: " +  str(nline_trn)
           + "; Current row count : "+ str(i*nBatch + nrows))

    # Drop ID and target
    y_train[i*nBatch:(i*nBatch+ np.min([nBatch, nrows])),0] = train['target']

    train.drop(['ID', 'target'], axis=1, inplace = True)

    # Proportio = (81, 82, 74, 65, 64, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 32, 33, 34, 35, 36, 37, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57) # white list
    #black_list = (214, 83, 84, 79, 80, 77, 78, 76, 69, 70, 71, 72, 68, 67, 66, 63, 62, 61, 60, 59, 58, 114, 73, 40, 41, 42, 43, 39, 38, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 21, 20, 19, 18, 17, 8, 9, 10, 11, 12, 75, 44, 404, 305, 283, 222, 202, 204, 216, 217, 466, 467, 493)
    #black_list += mixed_types
    print "\nDrop columns with most entities > 9990..."
    print "Drop columns with identical boolean and string variables "
    train.drop(all_nan.columns, axis = 1, inplace = True)
    train.drop(outlier_num.columns, axis=1, inplace = True)
    train.drop(outlier_bool.columns, axis=1, inplace = True)
    train.drop(outlier_str.columns, axis = 1, inplace = True)
    train.drop(outlier_frq_str.columns, axis = 1, inplace = True)
    print "Column count: %d"  % len(train.columns)
    #black_list_columns = [str(n).zfill(4) for n in black_list_columns]
    #black_list_columns = ['VAR_' + n for n in black_list_columns]  of NA values



    train_num = train.select_dtypes(include=[np.number])
    train_char = train.select_dtypes(include=[object])
    print("Numerical column count : "+ str(len(train_num.columns))+ "; Character column count : "+ str(len(train_char.columns)))

# It looks like NA is represented in character columns by -1 or [] or blank values, so convert these to explicit NAs. 
# Not entirely sure this is the right thing to do as there are real NA values, as well as -1 values already existing, 
# however it can be tested in predictive performance.
    print "Replace -1, [], empty as nan"
    train_char[train_char=="-1"] = NA
    train_char[train_char==""] = NA
    train_char[train_char=="[]"] = NA


    print "Parse the date"
    train_date = train[time_feature.columns]
    train_char.drop(time_feature.columns, axis = 1, inplace = True)
    
    print "Convert Boolean columns"
    list_diff = np.setdiff1d(train_char.columns, most_common_str.columns)    
    #boolen columns
    train_bool = train[list_diff]
    le = preprocessing.LabelEncoder()
    for c in list_diff:
       le.fit(train[c])
       train_bool.loc[:,c] = le.transform(train[c])

    print "Convert top 4 frequent string into indicator variables"
    train_dummy = train_char[most_common_str.columns]
    for c in most_common_str.columns:
        unique_list = most_common_str[c].dropna(axis = 0)
        temp = train_dummy[c]
        temp2 = temp.loc[~temp.isin(unique_list)]
        if len(temp2)> 0:
             train_dummy.loc[~temp.isin(unique_list),c] = NA
             temp_dummies = pd.get_dummies(train_dummy[c])
             train_dummy[c] = temp_dummies[unique_list[0]]
        for mm in np.arange(len(unique_list)):
             if mm > 0:
                  train_dummy[c+ '_'+str(mm)] = temp_dummies[unique_list[mm]]

# We place the date columns in a new dataframe and parse the dates
    for c in train_date.columns:
       train_date[c+ 'm']= (pd.to_datetime(train_date[c],format ='%d%b%y:%H:%M:%S').map(lambda x: x.month))
       train_date[c+ 'y']= (pd.to_datetime(train_date[c],format ='%d%b%y:%H:%M:%S').map(lambda x: x.year-2010))
       train_date[c]= (pd.to_datetime(train_date[c],format ='%d%b%y:%H:%M:%S').map(lambda x: (x.year-2010)*12+x.month))

# In[222]:
# Maintain the feature ordering
    print "Feature stored back...\n"
    data_cleaned = train
    data_cleaned[list_diff] = train_bool
#    data_cleaned[train_char.columns.values]= train_char
    data_cleaned[train_num.columns.values]= train_num
    data_cleaned[train_date.columns.values]= train_date
    data_cleaned[train_dummy.columns.values]  = train_dummy
    print("num of row: %d, num of col: %d" % (len(data_cleaned.index), len(data_cleaned.columns)))
    if i== 0:
        data_cleaned.to_csv('./data/train_cleaned.csv', index = False)
    else:
        data_cleaned.to_csv('./data/train_cleaned.csv', mode= 'a',
                             header = False, index = False)
    #data_cleaned = pd.concat([train_num, train_char, train_date])


   
Y = pd.DataFrame(y_train)
Y.columns = ['Target']
Y.to_csv('./data/train_label.csv', index= False)
#===================================================================

