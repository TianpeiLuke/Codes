# coding: utf-8

# Python script for Springleaf data preprocessing
# https://www.kaggle.com/c/springleaf-marketing-response
# modify to load the whole data

import numpy as np
from numpy import nan as NA
from pandas import Series, DataFrame
import pandas as pd
from sklearn import preprocessing

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
outlier_sparse = pd.read_csv('./data/outlier_sparse.csv')
# read the list of time featurs
time_feature = pd.read_csv('./data/list_time.csv')
most_common_str= pd.read_csv('./data/list_common_str4.csv')
summary_num =pd.read_csv('./data/list_summary_num.csv')

print "===================================================="
train = pd.read_csv("./data/train.csv")

nrows = len(train.index)
ncols  = len(train.columns)

print('Row count: %d' % nrows)
print('Column count: %d' % ncols)
print("Row count in total: " +  str(nline_trn))

y_train = train['target']
train.drop(['ID', 'target'], axis=1, inplace = True)


print "\nDrop all nan columns..." 
print "Drop columns with most entities > 9990..."
print "Drop columns with identical boolean and string variables "
train.drop(all_nan.columns, axis = 1, inplace = True)
train.drop(outlier_num.columns, axis=1, inplace = True)
train.drop(outlier_sparse.columns, axis= 1, inplace = True)
train.drop(outlier_bool.columns, axis=1, inplace = True)
train.drop(outlier_str.columns, axis = 1, inplace = True)
train.drop(outlier_frq_str.columns, axis = 1, inplace = True)
print "Column count: %d"  % len(train.columns)

train_num = train.select_dtypes(include=[np.number])
train_char = train.select_dtypes(include=[object])
print("Numerical column count : "+ str(len(train_num.columns))+ "; Character column count : "+ str(len(train_char.columns)))

print "Replace -1, [], empty as nan"
train_char[train_char=="-1"] = NA
train_char[train_char==""] = NA
train_char[train_char=="[]"] = NA

# Parse the dates into months, years, years+ months
print "Parse the date"
train_date = train[time_feature.columns]
train_char.drop(time_feature.columns, axis = 1, inplace = True)

for c in train_date.columns:
       train_date[c+ 'm']= (pd.to_datetime(train_date[c],format ='%d%b%y:%H:%M:%S').map(lambda x: x.month))
       train_date[c+ 'y']= (pd.to_datetime(train_date[c],format ='%d%b%y:%H:%M:%S').map(lambda x: x.year-2010))
       train_date[c]= (pd.to_datetime(train_date[c],format ='%d%b%y:%H:%M:%S').map(lambda x: (x.year-2010)*12+x.month))

# Convert the categorical data into numerical data
print "Convert Boolean columns"
list_diff = np.setdiff1d(train_char.columns, most_common_str.columns)
#boolen columns as the set difference (all char columns) - (string columnsi)
train_bool = train[list_diff]
le = preprocessing.LabelEncoder() #label encoder as 0,1
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

# Maintain the feature ordering
print "Feature stored back...\n"
data_cleaned = train
data_cleaned[list_diff] = train_bool
#data_cleaned[train_char.columns.values]= train_char
data_cleaned[train_num.columns.values] = train_num
data_cleaned[train_date.columns.values] = train_date
data_cleaned[train_dummy.columns.values]  = train_dummy
print("num of row: %d, num of col: %d" % (len(data_cleaned.index), len(data_cleaned.columns)))
data_cleaned.to_csv('./data/train_cleaned.csv', index = False)

Y = pd.DataFrame(y_train)
Y.columns = ['Target']
Y.to_csv('./data/train_label.csv', index= False)
#===================================================================


