# coding: utf-8

# Python script for Springleaf data preprocessing
# https://www.kaggle.com/c/springleaf-marketing-response
# modify to load the whole data
# This is for the test file

import numpy as np
from numpy import nan as NA
from pandas import Series, DataFrame
import pandas as pd
from sklearn import preprocessing

nline_trn = 145232 - 1
nline_tst = 145233 - 1
nBatch = 20000

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


print "===================================================="
test = pd.read_csv("./data/test.csv")

nrows = len(test.index)
ncols  = len(test.columns)

print('Row count: %d' % nrows)
print('Column count: %d' % ncols)
print("Row count in total: " +  str(nline_tst))
# no label to read


print "\nDrop all nan columns..." 
print "Drop columns with most entities > 9990..."
print "Drop columns with identical boolean and string variables "
test.drop(all_nan.columns, axis = 1, inplace = True)
test.drop(outlier_num.columns, axis=1, inplace = True)
test.drop(outlier_sparse.columns, axis = 1, inplace =True)
test.drop(outlier_bool.columns, axis=1, inplace = True)
test.drop(outlier_str.columns, axis = 1, inplace = True)
test.drop(outlier_frq_str.columns, axis = 1, inplace = True)
print "Column count: %d"  % len(test.columns)

test_num = test.select_dtypes(include=[np.number])
test_char = test.select_dtypes(include=[object])
print("Numerical column count : "+ str(len(test_num.columns))+ "; Character column count : "+ str(len(test_char.columns)))

print "Replace -1, [], empty as nan"
test_char[test_char=="-1"] = NA
test_char[test_char==""] = NA
test_char[test_char=="[]"] = NA

# Parse the dates, same as _trn.py version
print "Parse the date"
test_date = test[time_feature.columns]
test_char.drop(time_feature.columns, axis = 1, inplace = True)

for c in test_date.columns:
       test_date[c+ 'm']= (pd.to_datetime(test_date[c],format ='%d%b%y:%H:%M:%S').map(lambda x: x.month))
       test_date[c+ 'y']= (pd.to_datetime(test_date[c],format ='%d%b%y:%H:%M:%S').map(lambda x: x.year-2010))
       test_date[c]= (pd.to_datetime(test_date[c],format ='%d%b%y:%H:%M:%S').map(lambda x: (x.year-2010)*12+x.month))

# Convert the categorical data, same as _trn.py verison
print "Convert Boolean columns"
list_diff = np.setdiff1d(test_char.columns, most_common_str.columns)
#boolen columns
test_bool = test[list_diff]
le = preprocessing.LabelEncoder()
for c in list_diff:
   le.fit(test[c])
   test_bool.loc[:,c] = le.transform(test[c])

print "Convert top 4 frequent string into indicator variables"
test_dummy = test_char[most_common_str.columns]
for c in most_common_str.columns:
   unique_list = most_common_str[c].dropna(axis = 0)
   temp = test_dummy[c]
   temp2 = temp.loc[~temp.isin(unique_list)]     
   if len(temp2)> 0:
         test_dummy.loc[~temp.isin(unique_list),c] = NA
   temp_dummies = pd.get_dummies(test_dummy[c])
   test_dummy[c] = temp_dummies[unique_list[0]]
   for mm in np.arange(len(unique_list)):
       if mm > 0: 
            test_dummy[c+ '_'+str(mm)] = temp_dummies[unique_list[mm]]

# Maintain the feature ordering
print "Feature stored back...\n"
data_cleaned = test
data_cleaned[list_diff] = test_bool
data_cleaned[test_num.columns.values] = test_num
data_cleaned[test_date.columns.values] = test_date
data_cleaned[test_dummy.columns.values]  = test_dummy
print("num of row: %d, num of col: %d" % (len(data_cleaned.index), len(data_cleaned.columns)))
data_cleaned.to_csv('./data/test_cleaned.csv', index = False)

#===================================================================


