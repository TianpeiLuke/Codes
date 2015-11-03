%% use the real data set to do multiview learning

addpath('./libsvm-3.18/matlab/')
addpath('../../../../../../MATLAB/cvx/');

clearvars
close all
clc

% orig_folder = 'dataset';

% sub_folder = [{'a1a'},{'a2a'},{'a3a'},{'a4a'},{'a5a'},{'a6a'},{'a7a'}];
%% use the Page+ Link from WebKB dataset, provided by Vikas Sindhwani
load('PAGE.mat');
DataX{1} = X;
DataY = Y;
idxLabsTrn{1} = idxLabs;
load('LINK.mat');
DataX{2} = X;
%TrainY = Y;
idxLabsTrn{2} = idxLabs;

nV = 2;
nL = size(idxLabsTrn{1},2);
nU = size(DataX{1},1) - nL;
nTst = nU; % use the unlabeled set as test set
%% dataset and parameter setting
sigma = [1, 1];
sel = 2; % for initial SVM

% for mv-med
    param.regParam = 1; 
    for iv=1:nV
      param.kernelParm(iv)   = sqrt(1./(2*sigma(iv)^2));
      param.kernelMethod{iv} = {'rbf'};  
    end
    param.maxIterOut = 20;
    param.threOut = 1e-3;
    param.maxIterMAP = 50;
    param.threMAP = 1e-4;
    param.sigmaPri = 1.5;
    param.mode = 1;

  % for co-training
   param_ct.nsel = 3;
   param_ct.maxIterOut = 200;
   param_ct.psel = 3;
   param_ct.rpool = 0.5;
   param_ct.mode = 1;
   param_ct.Distribution = [{'mn'}, {'mn'}]; % multinomial for Bayes classifier
   
accuracy_v1_array = zeros( 1,size(idxLabs,1));
accuracy_v2_array = zeros( 1,size(idxLabs,1));
accuracy_mv_array = zeros( 1,size(idxLabs,1));
accuracy_ct_array = zeros( 1,size(idxLabs,1));



% cross validation
for R=1:size(idxLabs,1)
    display(sprintf('Repeat experiments %d:', R));
    
    optionsvm = cell(1,nV); 
    gamma_org = zeros(1,nV);
    
    Traindata.X_L{1} = DataX{1}(idxLabsTrn{1}(R,:),:);
    Traindata.X_L{2} = DataX{2}(idxLabsTrn{2}(R,:),:) ;
    Traindata.y_L = DataY(idxLabsTrn{1}(R,:)) ;
    
    Traindata.X_U{1} = DataX{1}( setdiff([1:size(DataX{1},1)],idxLabsTrn{1}(R,:)) ,:) ;
    Traindata.X_U{2} = DataX{2}( setdiff([1:size(DataX{2},1)],idxLabsTrn{2}(R,:)) ,:) ;
    Traindata.y_U = DataY(setdiff([1:size(DataX{1},1)],idxLabsTrn{1}(R,:))) ;
  
    Traindata.nU = nU;
    Traindata.nL = nL;
    Traindata.nV = nV;
    Traindata.d = [size(DataX{1},2),size(DataX{2},2)];

    Testdata.X_Tst{1} = DataX{1}( setdiff([1:size(DataX{1},1)],idxLabsTrn{1}(R,:)) ,:) ;
    Testdata.X_Tst{2} = DataX{2}( setdiff([1:size(DataX{2},1)],idxLabsTrn{2}(R,:)) ,:) ;
    Testdata.y_Tst = DataY(setdiff([1:size(DataX{1},1)],idxLabsTrn{1}(R,:))) ;
    Testdata.nTst  = nTst;
    Testdata.d = [size(DataX{1},2),size(DataX{2},2)];
    
%%  Train initial model 

    for iv=1:nV
      gamma_org(iv) = 1./(2*sigma(iv)^2);
      optionsvm{iv} = sprintf('-t %d -c 1 -g %f', sel ,gamma_org(iv));
    end
    
    model = cell(1,nV);
    accuracy_v = zeros(3,nV);
    % train two view independent classifiers y_L
    for iv = 1:nV
       model{iv} = svmtrain(Traindata.y_L, Traindata.X_L{iv},optionsvm{iv}); 
       
    [~, accuracy_v(:,iv), ~] = svmpredict(Testdata.y_Tst, Testdata.X_Tst{iv}, model{iv}); 
    end
  
    accuracy_v1_array(R) = accuracy_v(1,1);
    accuracy_v2_array(R) = accuracy_v(1,2);

    iniparam.inimodel = model;
    
%% Train with Mv-MED
[accuracy, errorlist, dev_tst, prob_tst, dev_trn, prob_trn,...
    history, history_tst] = mvmedbin(Traindata, Testdata, param, iniparam); 

accuracy_mv_array(R) = accuracy*100; 
    

%% Train with co-training

% before submitting, the zero-frequency feature should be tackled. i.e. 
% Add 1 to the count for every attribute value-class combination 
% (Laplace estimator) when an attribute value (Outlook=Overcast) doesn’t occur with every class value
Traindata_mod = Traindata;
Testdata_mod = Testdata;
epsilon_l = 1e-4;
% for iv=1:nV
%     TempXv1 = [Traindata.X_L{iv}(find(Traindata.y_L == 1),:)];
%     TempXv2 = [Traindata.X_L{iv}(find(Traindata.y_L == -1),:)];
%     TempX = TempX(:,any(TempX));
%     Traindata_mod.X_L{iv}=   round(Traindata_mod.X_L{iv}./epsilon_l+ ones(size(Traindata_mod.X_L{iv})));
%     Traindata_mod.X_U{iv}=   round(Traindata_mod.X_U{iv}./epsilon_l +ones(size(Traindata_mod.X_U{iv})));
%     Testdata_mod.X_Tst{iv} = round(Testdata_mod.X_Tst{iv}./epsilon_l+ ones(size(Testdata_mod.X_Tst{iv})));
%     
%     Traindata_mod.X_L{iv}(find(Traindata.y_L == 1),:)
%     Traindata_mod.X_L{iv}(find(Traindata.y_L == -1),:)
%     TempXv1 = [];
%     TempXv2 = [];
% end



[accuracy_ct, errorlist_ct, prob_tst_ct, Model_ct,...
    history_ct, history_tst_ct] = mv_cotrainingTWC(Traindata_mod, Testdata_mod, param_ct);

accuracy_ct_array(R) = accuracy_ct*100;   
   Traindata_mod = [];
   Testdata_mod = [];
end

display(sprintf('Accuracy v1: %.1f %%', mean(accuracy_v1_array)))
display(sprintf('Accuracy v2: %.1f %%', mean(accuracy_v2_array)))
display(sprintf('Accuracy MV-MED: %.1f %%', mean(accuracy_mv_array)))
display(sprintf('Accuracy co-training: %.1f %%', mean(accuracy_ct_array)))
