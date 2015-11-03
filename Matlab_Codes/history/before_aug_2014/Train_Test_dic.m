%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%% Load the data 
curpath = pwd; 
[dest_org, foldername, ext] = fileparts(curpath);

upper_org = '../../Raw_data/Features_data'; %'../../../Raw_data/Features_data';

choice = 1;

if choice == 1
%src_org =  strcat(upper_org, '/MFCC_win32ms');
src_org = strcat(upper_org, '/MFCC_win32ms_dim_red');
elseif choice == 2
%src_org =  strcat(upper_org, '/MFCC_win186ms');
src_org = strcat(upper_org, '/MFCC_win186ms_dim_red');
elseif choice ==3
%src_org =  strcat(upper_org, '/PLP_win186ms');
src_org = strcat(upper_org, '/PLP_win186ms_dim_red');  
elseif choice == 4
%src_org =  strcat(upper_org, '/MFCC_win32ms_envelop');
src_org = strcat(upper_org, '/MFCC_win32ms_envelop_dim_red'); 
else 
%src_org =  strcat(upper_org, '/PLP_envelop');
src_org = strcat(upper_org, '/PLP_envelop_dim_red');     
end

src_dir =  '/Dictionary/';

addpath('./libsvm-3.16/matlab/');
addpath('./SSC_ADMM_v1.1/')
addpath('./lapsvmp_v02/')
addpath('./SMRS/')


pathOrig = strcat(src_org, src_dir);

isload = 1;

if isload ==1
display('Loading the data...')
datasetname = strcat(pathOrig,'dataset_org.mat');
load(datasetname);
datasetname = strcat(pathOrig,'dataset_new.mat');
load(datasetname);
datasetname = strcat(pathOrig,'dataset_new_red.mat');
load(datasetname);    
end

testChan = [1:8, 10];

[D, N] = size(dataset_info);
dim = size(dataset_new{1},1);
%red_dim = size(dataset_new_red{1},1);

iscorrupt = zeros(D,N);
index     = zeros(D,N);
name      = zeros(D,N);
for i=1:D
   for j=1:N
       iscorrupt(i,j) = dataset_info(i,j).iscorrupted;
       index(i,j) = dataset_info(i,j).index;
       name_temp =   dataset_info(i,j).preName;
       name(i,j) = str2num(name_temp(7:8)); % DEC09 or DEC10
   end
end
clear dataset_org dataset_new ;
maxSub = max(index(1,:));

%%   splitting the data

%------------------------- class-wise index ------------------------
ind_c1 = find(class_label(1,:) == 1);
ind_c2 = find(class_label(1,:) == -1); 
ind_ag = intersect(find(iscorrupt(3,:)==0), find(iscorrupt(4,:)==0)); %all-good
ind_sb = union(find(iscorrupt(3,:)==1), find(iscorrupt(4,:)==1));     %some-bad

%--------------------- class-wise corruption index -------------------
 ind_c1_c = intersect(ind_c1, ind_sb);
 ind_c1_uc = intersect(ind_c1, ind_ag);
              % all-good part
 ind_c2_c = intersect(ind_c2, ind_sb);
 ind_c2_uc = intersect(ind_c2, ind_ag);
 
 
 %-------------------- class-wise dataset index ----------------------
 ind_c1_09 = intersect(ind_c1, find(name(1,:)==9));
 ind_c1_10 = intersect(ind_c1, find(name(1,:)==10));
 ind_c2_09 = intersect(ind_c2, find(name(1,:)==9));
 ind_c2_10 = intersect(ind_c2, find(name(1,:)==10));
 
 %---------------------class-wise dataset corruption index -----------
 ind_c1_09_c = intersect(ind_c1_09, ind_sb);
 ind_c1_09_uc = intersect(ind_c1_09, ind_ag);
 
 ind_c1_10_c = intersect(ind_c1_10, ind_sb);
 ind_c1_10_uc = intersect(ind_c1_10, ind_ag);
 
 ind_c2_09_c = intersect(ind_c2_09, ind_sb);
 ind_c2_09_uc = intersect(ind_c2_09, ind_ag);
 
 ind_c2_10_c = intersect(ind_c2_10, ind_sb);
 ind_c2_10_uc = intersect(ind_c2_10, ind_ag);
 
 %% --------------- Training set construction  ----------------------
 display('Training set construction: ')
 index_c1_temp = zeros(1,length(ind_c1));
 for k=1:length(ind_c1)
    index_c1_temp(k)  = dataset_info(1,ind_c1(k)).index;
 end
 index_c2_temp = zeros(1,length(ind_c2));
 for k=1:length(ind_c2)
    index_c2_temp(k)  = dataset_info(1,ind_c2(k)).index;
 end
 
 
 index_c1 = unique(index_c1_temp);
 index_c2 = unique(index_c2_temp);  % subject index for class 1 and class 2
 clear index_c1_temp index_c2_temp
 
 display('==========================================')
 nTrain_c1 = 40;
 nTrain_c2 = 40; % in subject
 display(sprintf('num of training subject (class 1): %d\n', nTrain_c1));
 display(sprintf('num of training subject (class 2): %d\n', nTrain_c2));
 
 % randomly select training subject in class 1
 Train_Index_c1 = index_c1(randsample(length(index_c1),nTrain_c1));
 display(['subject index for class 1: '])
 ind_train_c1 = [];
 for k= 1:length(Train_Index_c1)
  display(sprintf('%d \t',  Train_Index_c1(k)));   
  ind_train_c1 = [ind_train_c1, find(index(1,:) == Train_Index_c1(k))];
 end
 display('-----------------------');
 
 % randomly select training subject in class 2
 Train_Index_c2 = index_c2(randsample(length(index_c2),nTrain_c2));
 display(['subject index for class 2: '])
 ind_train_c2 = [];
 for k= 1:length(Train_Index_c2)
   display(sprintf('%d \t',  Train_Index_c2(k)));     
  ind_train_c2 = [ind_train_c2, find(index(1,:) == Train_Index_c2(k))];
 end
display('-----------------------');
%% ----------------- Test set construction

display('Testing set construction: ')
display(sprintf('num of testing subject (class 1): %d\n', length(index_c1)-nTrain_c1));
display(sprintf('num of testing subject (class 2): %d\n', length(index_c2)-nTrain_c2));
Test_Index_c1  = setdiff(index_c1,Train_Index_c1);
display(['subject index for class 1: '])
ind_test_c1 = [];
 for k= 1:length(Test_Index_c1)
  display(sprintf('%d \t',  Test_Index_c1(k)));   
  ind_test_c1 = [ind_test_c1, find(index(1,:) == Test_Index_c1(k))];
 end
 display('-----------------------');
 
Test_Index_c2  = setdiff(index_c2,Train_Index_c2);
display(['subject index for class 2: '])
ind_test_c2 = [];
 for k= 1:length(Test_Index_c2)
  display(sprintf('%d \t',  Test_Index_c2(k))); 
  ind_test_c2 = [ind_test_c2, find(index(1,:) == Test_Index_c2(k))];
 end
  display('-----------------------');
  
  
%% ---------------------------------------------  
save('./index_name_corrupt.mat', 'iscorrupt', 'name', 'index')
save('./info_spliting.mat', 'ind_c1', 'ind_c2', 'ind_ag',...
    'ind_sb', 'index_c1', 'index_c2');  
save('./Train_Test_sub_ind.mat', 'Train_Index_c1', 'ind_train_c1',...
    'Train_Index_c2', 'ind_train_c2',...
    'Test_Index_c1', 'ind_test_c1',...
    'Test_Index_c2', 'ind_test_c2');  
  
  
  