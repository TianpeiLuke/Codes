import pandas as pd
import numpy as np

##================================
# count the num of rows in file

import csv 

def rows_amount(filename):
    with open(filename) as f:
        for i, line in enumerate(f, 1):
            pass
    return i

orgdir = '../../../Documents/My_code/Springleaf-Kaggle/'
ddir = 'data'
train_filename = 'train.csv'
test_filename = 'test.csv'
filename = orgdir+ ddir+ '/'+ train_filename
print 'num of rows %d' %rows_amount(filename)
