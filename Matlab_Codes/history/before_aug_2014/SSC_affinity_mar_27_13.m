%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Build the affinity map by SSC
%        written by Tianpei Xie, 
%              Mar 26, 2013
%
%
%    SSC code from  Sparse Subspace Clustering: Algorithm, Theory, and Applications [ Code Available ]
%         E. Elhamifar, R. Vidal, to appear in IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2013.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

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
red_dim = size(dataset_new_red{1},1);

iscorrupt = zeros(D,N);
index     = zeros(D,N);
name      = zeros(D,N);
for i=1:D
   for j=1:N
       iscorrupt(i,j) = dataset_info(i,j).iscorrupted;
       index(i,j) = dataset_info(i,j).index;
       name_temp =   dataset_info(i,j).preName;
       name(i,j) = str2num(name_temp(7:8));
   end
end

ind_c1 = find(class_label(1,:) == 1);
ind_c2 = find(class_label(1,:) == -1); 

clear dataset_org;

%Y = dataset_new_red;

% for i=1:D
%    Y{i} = normc(Y{i}); %normalize the column 
% end
isload =1;

% compute SSC for each sensor
 C2 = cell(D,1);
 if isload == 0     
   for ii=1:D  
     C2{ii} = admmLasso_mat_func(dataset_new_red{ii}, true, 0.7);
   end
   save('C2-trial-2-unnorm.mat', C2);
 else 
     load('C2-trial-2-unnorm.mat');
 end
 
 % build the adjacency matrix
 CKSym = cell(D,1);
 CAbs  = cell(D,1);
 LapN  = cell(D,1);
 
 % construct the normalized Laplacian 
 for ii=1:D
  [CKSym{ii},CAbs{ii}] = BuildAdjacency(thrC(C2{ii},rho));
  DN = diag( 1./sqrt(sum(CKSym{ii})+eps) );
  LapN{ii} = speye(N) - DN * CKSym{ii} * DN;
 end




  
  



 










%% =================================================================
ifplot =0;
if ifplot == 1
    
  pcacoeff = cell(D,1);
  pca_proj = pcacoeff;

 for ii = 1:D
   pcacoeff{ii} = princomp(dataset_new{ii}');
   pca_proj{ii} = pcacoeff{ii}(:,1:red_dim);
 end

 ii=3;
 
 sub1 = 20;
 sub2 = 112;
 %---------------------
 ind_c1_c = intersect(ind_c1, find(iscorrupt(ii,:)==1));
 ind_c1_uc = intersect(ind_c1, find(iscorrupt(ii,:)==0));
 ind_c2_c = intersect(ind_c2, find(iscorrupt(ii,:)==1));
 ind_c2_uc = intersect(ind_c2, find(iscorrupt(ii,:)==0));
 
 %--------------------
 ind_indx_1 =  find(index(ii,:)==sub1);
 ind_indx_2 =  find(index(ii,:)==sub2);
 
 %--------------------
 ind_c1_09 = intersect(ind_c1, find(name(ii,:)==9));
 ind_c1_10 = intersect(ind_c1, find(name(ii,:)==10));
 ind_c2_09 = intersect(ind_c2, find(name(ii,:)==9));
 ind_c2_10 = intersect(ind_c2, find(name(ii,:)==10));
 
 %--------------
 ind_c1_09_c = intersect(ind_c1_09, find(iscorrupt(ii,:)==1));
 ind_c1_09_uc = intersect(ind_c1_09, find(iscorrupt(ii,:)==0));
 
 ind_c1_10_c = intersect(ind_c1_10, find(iscorrupt(ii,:)==1));
 ind_c1_10_uc = intersect(ind_c1_10, find(iscorrupt(ii,:)==0));
 
 ind_c2_09_c = intersect(ind_c2_09, find(iscorrupt(ii,:)==1));
 ind_c2_09_uc = intersect(ind_c2_09, find(iscorrupt(ii,:)==0));
 
 ind_c2_10_c = intersect(ind_c2_10, find(iscorrupt(ii,:)==1));
 ind_c2_10_uc = intersect(ind_c2_10, find(iscorrupt(ii,:)==0));
 
 %---------------
 
 
 
 %========================
   figure(3);
   show_dataset = dataset_new_red{ii};
   proj_mean_1  = pca_proj{ii}'*mean_class{ii,1};
   proj_mean_2  = pca_proj{ii}'*mean_class{ii,2};
   show_dataset(:,ind_c1)= show_dataset(:,ind_c1) + repmat(proj_mean_1,1,length(ind_c1));
   show_dataset(:,ind_c2)= show_dataset(:,ind_c2) + repmat(proj_mean_2,1,length(ind_c2));
   
   plot(show_dataset(1,ind_c1), show_dataset(2,ind_c1), 'or', 'Linewidth',1.5);
   grid on;
   hold on;
   plot(show_dataset(1,ind_c2), show_dataset(2,ind_c2), 'xb');
   hold off;
   xlabel('1st pca');
   ylabel('2nd pca'); 
   if ismember(choice, [1,2,4]) 
      type = 'mfcc';
     
   else
       type = 'plp';
   end
    title(sprintf('reduced feature (%s) sensor %d', type, testChan(ii)));
  legend('human', 'human-animal')

  %============================
  figure (4);
   plot(show_dataset(1,ind_c1_uc), show_dataset(2,ind_c1_uc), 'or', 'Linewidth',1);
   hold on;
   plot(show_dataset(1,ind_c2_uc), show_dataset(2,ind_c2_uc), 'xb', 'Linewidth',1);
   hold on;
   plot(show_dataset(1,ind_c1_c), show_dataset(2,ind_c1_c), '+m', 'Linewidth',2);
   grid on;
   hold on; 
   plot(show_dataset(1,ind_c2_c), show_dataset(2,ind_c2_c), '+c', 'Linewidth',2); 
   hold off;
   xlabel('1st pca');
   ylabel('2nd pca'); 
   if ismember(choice, [1,2,4]) 
      type = 'mfcc';
     
   else
       type = 'plp';
   end
    title(sprintf('reduced feature (%s) sensor %d', type, testChan(ii)));
  legend('human-uncorrupted', 'human-animal-uncorrupted',...
      'human-corrupted','human-animal-corrupted')
  
  %===============================
  figure(5);
  plot(show_dataset(1,ind_c1_uc), show_dataset(2,ind_c1_uc), 'oy', 'Linewidth',1);
   hold on;
    grid on;
   plot(show_dataset(1,ind_c2_uc), show_dataset(2,ind_c2_uc), 'xc', 'Linewidth',1);
   hold on;
  plot(show_dataset(1,ind_indx_1), show_dataset(2,ind_indx_1), '+r', 'Linewidth',2.5);
   hold on;
    plot(show_dataset(1,ind_indx_2), show_dataset(2,ind_indx_2), '+b', 'Linewidth',2.5);
hold off;
   xlabel('1st pca');
   ylabel('2nd pca');
   title(sprintf('reduced feature (%s) sensor %d', type, testChan(ii)));
  legend('human-uncorrupted', 'human-animal-uncorrupted',...
      ['subject ' num2str(sub1)],['subject ' num2str(sub2)])
  
  
  %========================================
  figure(6);
  plot(show_dataset(1,ind_c1_09), show_dataset(2,ind_c1_09), 'om', 'Linewidth',2);
   hold on;
   plot(show_dataset(1,ind_c2_09), show_dataset(2,ind_c2_09), 'xc', 'Linewidth',2);
   hold on;
  plot(show_dataset(1,ind_c1_10), show_dataset(2,ind_c1_10), 'or', 'Linewidth',2.5);
   hold on;
    grid on;
    plot(show_dataset(1,ind_c2_10), show_dataset(2,ind_c2_10), 'xb', 'Linewidth',2.5);
hold off;
   xlabel('1st pca');
   ylabel('2nd pca');
   title(sprintf('reduced feature (%s) sensor %d', type, testChan(ii)));
  legend('human-09', 'human-animal-09',...
      'human-10', 'human-animal-10')
  
   figure(7);
  plot(show_dataset(1,ind_c1_09), show_dataset(2,ind_c1_09), 'om', 'Linewidth',2);
   hold on;
   plot(show_dataset(1,ind_c2_09), show_dataset(2,ind_c2_09), 'xc', 'Linewidth',2);
   grid on;
hold off;
   xlabel('1st pca');
   ylabel('2nd pca');
   title(sprintf('reduced feature (%s) sensor %d', type, testChan(ii)));
  legend('human-09', 'human-animal-09')
  
   figure(8);
  plot(show_dataset(1,ind_c1_10), show_dataset(2,ind_c1_10), 'or', 'Linewidth',2);
   hold on;
   grid on;
    plot(show_dataset(1,ind_c2_10), show_dataset(2,ind_c2_10), 'xb', 'Linewidth',2);
hold off;
   xlabel('1st pca');
   ylabel('2nd pca');
   title(sprintf('reduced feature (%s) sensor %d', type, testChan(ii)));
  legend('human-10', 'human-animal-10')
  
  
  %=========================================
  
   figure(6);
  plot(show_dataset(1,ind_c1_09_uc), show_dataset(2,ind_c1_09_uc), 'om', 'Linewidth',2);
   hold on;
   plot(show_dataset(1,ind_c2_09_uc), show_dataset(2,ind_c2_09_uc), 'xc', 'Linewidth',2);
   hold on;
  plot(show_dataset(1,ind_c1_10_uc), show_dataset(2,ind_c1_10_uc), 'or', 'Linewidth',2);
   hold on;
  plot(show_dataset(1,ind_c2_10_uc), show_dataset(2,ind_c2_10_uc), 'xb', 'Linewidth',2);
   hold on 
  plot(show_dataset(1,ind_c1_09_c), show_dataset(2,ind_c1_09_c), '+y', 'Linewidth',3);
   hold on;
  plot(show_dataset(1,ind_c2_09_c), show_dataset(2,ind_c2_09_c), '*g', 'Linewidth',3);
   hold on;
  plot(show_dataset(1,ind_c1_10_c), show_dataset(2,ind_c1_10_c), '+k', 'Linewidth',2);
   hold on;
  plot(show_dataset(1,ind_c2_10_c), show_dataset(2,ind_c2_10_c), '*y', 'Linewidth',2);
   hold off;
   grid on;
   xlabel('1st pca');
   ylabel('2nd pca');
   title(sprintf('reduced feature (%s) sensor %d', type, testChan(ii)));
  legend('human-09-uncorrupted', 'human-animal-09-uncorrupted',...
      'human-10-uncorrupted', 'human-animal-10-uncorrupted',... 
      'human-09-corrupted', 'human-animal-09-corrupted',...
      'human-10-corrupted', 'human-animal-10-corrupted')
  
   figure(7);
  plot(show_dataset(1,ind_c1_09_uc), show_dataset(2,ind_c1_09_uc), 'om', 'Linewidth',2);
   hold on;
   plot(show_dataset(1,ind_c2_09_uc), show_dataset(2,ind_c2_09_uc), 'xc', 'Linewidth',2);
  plot(show_dataset(1,ind_c1_09_c), show_dataset(2,ind_c1_09_c), '+y', 'Linewidth',3);
  plot(show_dataset(1,ind_c2_09_c), show_dataset(2,ind_c2_09_c), '*g', 'Linewidth',3);
   grid on;
hold off;
   xlabel('1st pca');
   ylabel('2nd pca');
   title(sprintf('reduced feature (%s) sensor %d', type, testChan(ii)));
  legend('human-09-uncorrupted', 'human-animal-09-uncorrupted',...
      'human-09-corrupted', 'human-animal-09-corrupted')
  
   figure(8);
  plot(show_dataset(1,ind_c1_10_uc), show_dataset(2,ind_c1_10_uc), 'or', 'Linewidth',2);
   hold on;
   grid on;
    plot(show_dataset(1,ind_c2_10_uc), show_dataset(2,ind_c2_10_uc), 'xb', 'Linewidth',2);
    plot(show_dataset(1,ind_c1_10_c), show_dataset(2,ind_c1_10_c), '+k', 'Linewidth',2);
    plot(show_dataset(1,ind_c2_10_c), show_dataset(2,ind_c2_10_c), '*c', 'Linewidth',2);
hold off;
   xlabel('1st pca');
   ylabel('2nd pca');
   title(sprintf('reduced feature (%s) sensor %d', type, testChan(ii)));
  legend('human-10-uncorrupted', 'human-animal-10-uncorrupted',...
      'human-10-corrupted', 'human-animal-10-corrupted')
  
  
  %===========================
  figure(9);
  ind_c1_09_uc_rep = intersect(ind_c1_09_uc, find(ind_rep==1));
  ind_c2_09_uc_rep = intersect(ind_c2_09_uc, find(ind_rep==1));
  ind_c1_10_uc_rep = intersect(ind_c1_10_uc, find(ind_rep==1));
  ind_c2_10_uc_rep = intersect(ind_c2_10_uc, find(ind_rep==1));
  ind_c1_09_c_rep = intersect(ind_c1_09_c, find(ind_rep==1));
  ind_c2_09_c_rep = intersect(ind_c2_09_c, find(ind_rep==1));
  ind_c1_10_c_rep = intersect(ind_c1_10_c, find(ind_rep==1));
  ind_c2_10_c_rep = intersect(ind_c2_10_c, find(ind_rep==1));
  
    
  
  
  plot(show_dataset(1,ind_c1_09_uc_rep), show_dataset(2,ind_c1_09_uc_rep), 'om', 'Linewidth',2);
   hold on;
   plot(show_dataset(1,ind_c2_09_uc_rep), show_dataset(2,ind_c2_09_uc_rep), 'xc', 'Linewidth',2);
   hold on;
  plot(show_dataset(1,ind_c1_10_uc_rep), show_dataset(2,ind_c1_10_uc_rep), 'or', 'Linewidth',2);
   hold on;
  plot(show_dataset(1,ind_c2_10_uc_rep), show_dataset(2,ind_c2_10_uc_rep), 'xb', 'Linewidth',2);
   hold on 
  plot(show_dataset(1,ind_c1_09_c_rep), show_dataset(2,ind_c1_09_c_rep), '+y', 'Linewidth',3);
   hold on;
  plot(show_dataset(1,ind_c2_09_c_rep), show_dataset(2,ind_c2_09_c_rep), '*g', 'Linewidth',3);
   hold on;
  plot(show_dataset(1,ind_c1_10_c_rep), show_dataset(2,ind_c1_10_c_rep), '+k', 'Linewidth',2);
   hold on;
  plot(show_dataset(1,ind_c2_10_c_rep), show_dataset(2,ind_c2_10_c_rep), '*y', 'Linewidth',2);
   hold off;
   grid on;
   xlabel('1st pca');
   ylabel('2nd pca');
   title(sprintf('reduced feature (%s) sensor %d', type, testChan(ii)));
  legend('human-09-uncorrupted', 'human-animal-09-uncorrupted',...
      'human-10-uncorrupted', 'human-animal-10-uncorrupted',... 
      'human-09-corrupted', 'human-animal-09-corrupted',...
      'human-10-corrupted', 'human-animal-10-corrupted')
  axis([55,95,-30,5])
  
  
   figure(7+3);
  plot(show_dataset(1,ind_c1_09_uc_rep), show_dataset(2,ind_c1_09_uc_rep), 'om', 'Linewidth',2);
   hold on;
   plot(show_dataset(1,ind_c2_09_uc_rep), show_dataset(2,ind_c2_09_uc_rep), 'xc', 'Linewidth',2);
  plot(show_dataset(1,ind_c1_09_c_rep), show_dataset(2,ind_c1_09_c_rep), '+y', 'Linewidth',3);
  plot(show_dataset(1,ind_c2_09_c_rep), show_dataset(2,ind_c2_09_c_rep), '*g', 'Linewidth',3);
   grid on;
hold off;
   xlabel('1st pca');
   ylabel('2nd pca');
   title(sprintf('reduced feature (%s) sensor %d', type, testChan(ii)));
  legend('human-09-uncorrupted', 'human-animal-09-uncorrupted',...
      'human-09-corrupted', 'human-animal-09-corrupted')
  
   figure(8+3);
  plot(show_dataset(1,ind_c1_10_uc_rep), show_dataset(2,ind_c1_10_uc_rep), 'or', 'Linewidth',2);
   hold on;
   grid on;
    plot(show_dataset(1,ind_c2_10_uc_rep), show_dataset(2,ind_c2_10_uc_rep), 'xb', 'Linewidth',2);
    plot(show_dataset(1,ind_c1_10_c_rep), show_dataset(2,ind_c1_10_c_rep), '+k', 'Linewidth',2);
    plot(show_dataset(1,ind_c2_10_c_rep), show_dataset(2,ind_c2_10_c_rep), '*c', 'Linewidth',2);
hold off;
   xlabel('1st pca');
   ylabel('2nd pca');
   title(sprintf('reduced feature (%s) sensor %d', type, testChan(ii)));
  legend('human-10-uncorrupted', 'human-animal-10-uncorrupted',...
      'human-10-corrupted', 'human-animal-10-corrupted')
  %axis([55,95,-30,5])
  
  %==============================================================
  figure(12)
 
 
  C2_show_11 = zeros(size(CKSym{ii}));
  C2_show_22 = zeros(size(CKSym{ii}));
  C2_show_12 = zeros(size(CKSym{ii}));
 
  C_show = C2_show_11;
  ind_rand = randsample(N, 1000);
  ind_rand_c1 = intersect(ind_rand, ind_c1);
  ind_rand_c2 = intersect(ind_rand, ind_c2);
  
  
   C2_show_11(ind_rand_c1,ind_rand_c1) = CKSym(ind_rand_c1,ind_rand_c1);
  C2_show_22(ind_rand_c2,ind_rand_c2) = CKSym(ind_rand_c2,ind_rand_c2);
  C2_show_12(ind_rand_c1,ind_rand_c2) = CKSym(ind_rand_c1,ind_rand_c2);
  C2_show_12(ind_rand_c2,ind_rand_c1) = CKSym(ind_rand_c2,ind_rand_c1);
  
    figure(12)
  
    gplot(C2_show_11,dataset_new_red{ii}(1:2,:)','-ob')
    hold on;
    gplot(C2_show_22,dataset_new_red{ii}(1:2,:)','-xr')
    gplot(C2_show_12,dataset_new_red{ii}(1:2,:)','-*c')
   hold off;
   grid on;
   axis([60,100,-30,5])
end