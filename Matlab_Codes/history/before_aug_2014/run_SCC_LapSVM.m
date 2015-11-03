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
addpath('./lapsvmp_v02/classifiers/')
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
red_dim = size(dataset_new_red{1},1);
clear dataset_org dataset_new ;


iscorrupt = zeros(D,N);
index     = zeros(D,N);
name      = zeros(D,N);

display('Loading the data information and training/testing splitting...')
load('./index_name_corrupt.mat')
load('./info_spliting.mat');  
load('./Train_Test_sub_ind.mat');  

Trainset = cell(D,1);
Testset = cell(D,1);
Y       = cell(D,1);

for ii = 1:D
   Trainset{ii} =  dataset_new_red{ii}(:,union(ind_train_c1,ind_train_c2));
   Testset{ii} =  dataset_new_red{ii}(:,union(ind_test_c1,ind_test_c2));
   Y{ii} = Trainset{ii}; %[Trainset{ii}, Testset{ii}];   
end
Train_label = class_label(1,union(ind_train_c1,ind_train_c2));
Test_label = class_label(1,union(ind_test_c1,ind_test_c2));
%% Construct the Laplacian map
 

[LapN,LapUN, SpC, CKSym, CAbs]= SSC_partial(Y, true, 20, 1 , 5);
display('Saving data...')
save('Laplacian.mat', 'LapN', 'SpC', 'CKSym', 'CAbs' );
%% Build the Heat kernel for all data point
options.Kernel =  'rbf'; 
options.KernelParam =  0.35*100;
Kernel = cell(D,1);
display('Build Kernel');
for ii=1:D
%    Kernel{ii} = zeros(N,N);
%    for n=1:N
%        for m=1:N   
%          Kernel{ii}(n,m) = exp(-norm(dataset_new_red{ii}(:,n)...
%                                 - dataset_new_red{ii}(:,m))/gamma);
%        end
%    end
 %data_temp = [Trainset{ii}'; Testset{ii}'];
 Kernel{ii} = calckernel(options, Y{ii}');
 %data_temp = [];
end

%% Call LapSVM
classifer_svm = cell(D,1);
newKernel = cell(D,1);
classifer_alpha = cell(D,1);
classifer_b     = cell(D,1);
predict_label   = cell(D,1);
accuracy = zeros(D,1);
accuracy_h = zeros(D,1);
accuracy_ha = zeros(D,1);

for ii=1:D
 display(['Train LapSVM for sensor ' num2str(testChan(ii))]);   
    %-------- parameter setting --------------
 data.X = Y{ii}'; 
 data.Y = Train_label'; %[class_label(ii, union(ind_train_c1,ind_train_c2))';...
                          %zeros(length(union(ind_test_c1,ind_test_c2)),1)];
 data.K = Kernel{ii};
 data.L = LapUN{ii};
 
 options.gamma_A=0.014362;
 options.gamma_I=0.7852;
 
 options.Cg=1;    % Conjugate Gradient method      
 options.CgStopType=0;
 options.MaxIter=200;
 options.Verbose=0;
 options.UseBias = 1;
 options.LaplacianNormalize = 0;
 %--------- train LapSVM ----------------------
 classifer_svm{ii} =  lapsvmp(options,data);
 classifer_alpha{ii} = classifer_svm{ii}.alpha;
 classifer_b{ii} = classifer_svm{ii}.b;
 
 display(['Testing LapSVM for sensor ' num2str(testChan(ii))]);   
 % ---------prediction ------------------------
 newKernel{ii} = calckernel(options, classifer_svm{ii}.xtrain, Testset{ii}');
 
 z=real(newKernel{ii}*classifer_alpha{ii} +  classifer_b{ii});
 z(z>1)=1;
 z(z<-1)=-1;
 predict_label{ii} = sign(z);
 accuracy(ii) = length(find(predict_label{ii}==Test_label'))/length(Test_label);
 accuracy_h(ii) = length(intersect(find(predict_label{ii}==Test_label'),...
                                    find(Test_label'== 1)))/length(find(Test_label'== 1));
 accuracy_ha(ii) = length(intersect(find(predict_label{ii}==Test_label'),...
                                    find(Test_label'== -1)))/length(find(Test_label'== -1));                               
 display(sprintf('Test Overall Accuracy: %3.1f %% \n', accuracy(ii)*100))
 display(sprintf('Test Human Accuracy: %3.1f %% \n', accuracy_h(ii)*100))
 display(sprintf('Test Human-animal Accuracy: %3.1f %% \n', accuracy_ha(ii)*100))
end


accuracy
accuracy_h
accuracy_ha
save('Accuracy.mat', 'accuracy', 'accuracy_h', 'accuracy_ha');



