%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  This code is to redo Nam's work based on his code
%
% {Reminder}:  change the name and note for every test with syn on server
%
% by Tianpei Xie, Oct_31_2012
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

curpath = pwd; 
[dest_org, foldername, ext] = fileparts(curpath);
%upper_org =  '../../../Raw_data/Segmented_data';

upper_org2 = '../../Raw_data/Features_data';  %'../../../Raw_data/Features_data';
%src_org =  strcat(upper_org, '/Seg_70per_overlap');

%choice = 1 for cepstrum, =2 for MFCC , =3 for log-scale cepstrum

choice = 3;

if choice ==1 
   src_org = strcat(upper_org2, '/Ceps_70per_overlap');
elseif choice ==2
   src_org = strcat(upper_org2,'/MFCC_70per_overlap'); 
elseif choice ==3
   src_org = strcat(upper_org2, '/Spec_70per_overlap');
end

   
src_dir = [{'/Dictionary/'}];   
  
      
testChan  = [1, 2, 4]; %[1 2 3 4 ];
chanCode = [];
for k=1:length(testChan)
    chanCode = strcat(chanCode, sprintf('%d',testChan(k)));
end

%% Parameter setting for dictionary 


fs = 10000;
%NmSubPerDictTrain   = 34;  % num of subject for each train dict
%NmSubPerDictTest    = 15;  % .. . .              ... test dict
%NmSamplePerSubTrain = 20;  % num of sample for each subject in traing dict
%NmSamplePerSubTest  = 5;   % ... ..            

pathOrig = strcat(src_org, src_dir{1});
%pathDest = strcat(dest_org, dest_dir{1});
%pathOrig = pathDest;



%% load the dictionary directly 

sel = 1;  % sel = 1, train10 test09; sel==2, train09 test10

%% load the dictionary directly 
display('Loading... ') 
if sel==1
  train_filename = 'Dceps_train_10';
else
  train_filename = 'Dceps_train_09';  
end
if choice == 1
     train_filename = strcat('Cep_',train_filename, '_Chan', chanCode);
elseif choice ==2
     train_filename = strcat('Mfcc_',train_filename,'_Chan', chanCode);
else
     train_filename = strcat('Spec_',train_filename,'_Chan', chanCode);
end

display(['Load to ' train_filename]);
load(strcat(pathOrig,train_filename, '.mat')); 

if sel==1
  test_filename = 'Dceps_test_09';
else
  test_filename = 'Dceps_test_10';
end

if choice == 1
     test_filename = strcat('Cep_',test_filename,'_Chan',chanCode);
elseif choice ==2
     test_filename = strcat('Mfcc_',test_filename,'_Chan',chanCode);
else
     test_filename = strcat('Spec_',test_filename,'_Chan',chanCode);
end

display(['Load to ' test_filename]);
load(strcat(pathOrig,test_filename, '.mat'));

% Note that 
% Index_system{1} = preName_list;
% Index_system{2} = sufName_list;
% Index_system{3} = subindex_list;
% Index_system{4} = ishuman_list;
% Index_system{5} = iscorrupt_list;
% Index_system{6} = dataset_list;

index_filename = 'global_index_system';
load(strcat(pathOrig, index_filename, '.mat'));
preName_list = Index_system{1};
sufName_list = Index_system{2};
subindex_list=  Index_system{3};
ishuman_list = Index_system{4}; 
iscorrupt_list =Index_system{5};
dataset_list = Index_system{6};
%============================================================
%% Auxilary parameter setting
nDataset = length(Test_metafilelist);                 % # of dataset
nTotalTestSub = size(Test_metafilelist(1).info{1},2); % # of subject
rec_error  = cell(nDataset, nTotalTestSub);           % record of recon error
label_record = zeros(nDataset, nTotalTestSub);        % record of ground truth
subject_record = zeros(nDataset, nTotalTestSub);      % record of subject number
corrupt_record = zeros(nDataset, nTotalTestSub);      % record of corrupt index
error      = zeros(nDataset, nTotalTestSub);          % record of decision error
vDec       = zeros(nDataset, nTotalTestSub);          % record of decision
nChan = length(Test_metafilelist(1).dict);            %num of channel of concern

% record of ground truth (label) and subject index
test_label = zeros(nDataset,nTotalTestSub*NmSamplePerSubTest);
test_subNum = zeros(nDataset ,nTotalTestSub*NmSamplePerSubTest);
test_corrupt = zeros(nDataset ,nTotalTestSub*NmSamplePerSubTest);

% record of sparse code
W_record   = cell(nDataset,nTotalTestSub );
W_info_record = cell(nDataset,nTotalTestSub );
sigma_record  = cell(nDataset,nTotalTestSub);


% record of human, human-animal, overall acurracy 
h_acc   = zeros(nDataset, 1);
ha_acc = zeros(nDataset,1);
overall_acc = zeros(nDataset,1);

for nn = 1: min([nDataset,3]);
  train_h_ha = Train_metafilelist(nn).dict; 
  n_train_h = Train_metafilelist(nn).nh;
  test_h_ha = Test_metafilelist(nn).dict; 
  n_test_h = Test_metafilelist(nn).nh;

  nSampleTrain = size(train_h_ha{1},2);
  nSampleTest = size(test_h_ha{1},2);
  

  maxTrain = zeros(nChan,1);
  maxTest  = zeros(nChan,1);
 for i=1:nChan   
    maxTrain(i) =max(max(abs(train_h_ha{i})));
    maxTest(i) =max(max(abs(test_h_ha{i})));
 end
 for i=1:nChan
    train_h_ha{i} = train_h_ha{i}/max([maxTrain(i), maxTest(i)]);
    test_h_ha{i} = test_h_ha{i}/max([maxTrain(i), maxTest(i)]);
 end
%% confine the atoms on the unit ball for cepstrum 

if choice ==1 || choice == 3
   for i=1:nChan 
    for k=1:nSampleTrain
       train_h_ha{i}(:,k) = train_h_ha{i}(:,k) - mean(train_h_ha{i}(:,k));
       train_h_ha{i}(:,k) = train_h_ha{i}(:,k)/norm(train_h_ha{i}(:,k));
    end
    for k=1:nSampleTest
       test_h_ha{i}(:,k) = test_h_ha{i}(:,k) - mean(test_h_ha{i}(:,k));
       test_h_ha{i}(:,k) = test_h_ha{i}(:,k)/norm(test_h_ha{i}(:,k));
    end
   end
end

%% obtain the ground truth
test_info = Test_metafilelist(nn).info{1};
for i=1:nSampleTest
    id_sub = floor((i-1)/NmSamplePerSubTest)+1;
    id_sample = mod((i-1), NmSamplePerSubTest)+1;
    temp1 = test_info{id_sub};
    temp2 = temp1(id_sample);
    test_label(nn,i) =  temp2.ishuman;
    test_subNum(nn,i) = temp2.subject;
    test_corrupt(nn,i) = temp2.iscorrupt;
end


%% Loop for implemetation
display('Parameter for reweighed MTMV');
opts.lambda = 120;
opts.gamma_e = 1e-3;                   %initial value for epsilon -> 0
epsilon = 1e-2; 
opts.maxIter = 400;
opts.thres = 8e-3;%2e-2;
opts.sigma = ones(1,nChan);            %confidence for each channels
opts.sigma = opts.sigma/norm(opts.sigma,1);
opts.tao   = 10;
opts.isfixed = 0;
opts
%=================================================================
% {Reminder:} change the filename for every test if opts changes
%=================================================================
test_batch = cell(nChan,1);
perm_test  = randperm(nTotalTestSub);


  for id_sub = 1:nTotalTestSub%test on each subject
    % for each test subject, we select s=NmSamplePerSubTest; to be a batch
    display(['Subject: ' num2str(test_subNum(nn, (perm_test(id_sub)-1)*NmSamplePerSubTest+1))]);
    % store the subject number
    subject_record(nn, id_sub) = test_subNum(nn, (perm_test(id_sub)-1)*NmSamplePerSubTest+1);
    corrupt_record(nn, id_sub) = test_corrupt(nn, (perm_test(id_sub)-1)*NmSamplePerSubTest+1);

    ind_batch = (perm_test(id_sub)-1)*NmSamplePerSubTest + [1:NmSamplePerSubTest];
    label = unique(test_label(nn,ind_batch));
    label_record(nn, id_sub) = label;  % store the ground truth
    
    for i=1:nChan
        test_batch{i} = test_h_ha{i}(:, ind_batch);
    end
    %% MTMV lasso
    display('Start algorithm...');
    t3 = cputime;
    %[W,iter] = alm_mTasks_mVariates(train_h_ha, test_batch, opts);
    [W, iter, info] = reweight_mTasks_mVariates_v3(train_h_ha, test_batch, opts);
    t4 = cputime;
    display('End algorithm.')
    W_record{nn,id_sub} = W;
    W_info_record{nn,id_sub} = info;
    sigma_record{nn,id_sub} = info.sigma;

    sprintf('Total time for sparse coding = %f', t4 - t3);
    %% Test result
    err1 = 0; err2 = 0; % reconstruction error
    partition = [{[1:n_train_h]}; {n_train_h+1:nSampleTrain}];
    % a partition of the sparse code 
    
    % Compute errors
    for i = 1:nChan
       range = (i-1)*NmSamplePerSubTest+ [1:NmSamplePerSubTest];
       err1 = err1 +  norm(test_batch{i} - ...
                      train_h_ha{i}(:,partition{1})*W(partition{1},range) , 'fro')^2; 
%            err1 = err1 + norm(Y{nC} - D_h_train{chanIndx(nC)}(feaSel,trainSamples)*W(1:nh_train,(nC-1)*nSegs+1 : nC*nSegs ), 'fro'  )^2;   % for training date 09       
       err2 = err2 + norm(test_batch{i} - train_h_ha{i}(:,partition{2})*W(partition{2}, range ), 'fro' )^2; 
    end
    rec_error{nn, id_sub} = [err1, err2]; 
    
    
    % make decision on minimal recon errors
    if err1 <= err2  vDec(nn,id_sub) = 1; %if err1<err2 , ishuman =1
    else vDec(nn,id_sub) = 0;
    end
    
    if vDec(nn,id_sub) == label
        error(nn,id_sub) =0;
    else
        error(nn,id_sub) =1;
    end
    
  end
end

for nn=1:nDataset
h_acc(nn) = 1 - sum(error(nn,1:NmSubPerDictTest_h))/NmSubPerDictTest_h;
ha_acc(nn) = 1- sum(error(nn,NmSubPerDictTest_h+1:nTotalTestSub))/NmSubPerDictTest_ha;
overall_acc(nn) = 1- sum(error(nn,:))/(nTotalTestSub);
display(sprintf('Human footstep accuracy = %f %%', h_acc(nn)*100));
display(sprintf('Human-animal footstep accuracy = %f %%', ha_acc(nn)*100));
display(sprintf('Overall accuracy = %f %%', overall_acc(nn)*100));
end
Note = 'Reweighted Least square, test on 3 randomly selected dataset with DEC 10 as training set and DEC 09 as testing, data are corrupted manually, tao=10 ';

mean_h_acc = mean(h_acc);
var_h_acc = sqrt(var(h_acc));
mean_ha_acc = mean(ha_acc);
var_ha_acc = sqrt(var(ha_acc));
mean_overall_acc = mean(overall_acc);
var_overall_acc = sqrt(var(overall_acc));

trial = 21;

save(strcat('decVec_cepsFeas_RW_train10','_Ndataset_',num2str(nDataset),'_trial_', num2str(trial), '.mat'), 'opts', ...
     'h_acc', 'ha_acc', 'overall_acc', 'vDec','error', ...
     'mean_h_acc', 'mean_ha_acc', 'mean_overall_acc',...
      'var_h_acc',  'var_ha_acc',  'var_overall_acc',...
    'Note', 'W_record', 'W_info_record, ''nDataset', ...
     'test_info', 'Train_metafilelist', 'Test_metafilelist', ...  
     'n_train_h', 'n_test_h',...
     'rec_error', 'label_record','subject_record', 'corrupt_record',  ...
     'nSampleTest', 'nSampleTrain',...
     'testChan',...
     'NmSubPerDictTrain','NmSubPerDictTest_h', 'NmSubPerDictTest_ha', 'nTotalTestSub',...
     'NmSamplePerSubTrain','NmSamplePerSubTest' );


