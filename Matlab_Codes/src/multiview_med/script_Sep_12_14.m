%% Script to test mvmedbin.m 
addpath('./libsvm-3.18/matlab/')
addpath('../../../../../../MATLAB/cvx/');
%clear all
clearvars
close all
clc


% mu_111 = [-1.1, 1];
% mu_112 = [-1.1, 1];
% 
% mu_121 = [1, -1.1];
% mu_122 = [1, -1.1];
% 
% mu_211 = [1.2, 1.2];
% mu_212 = [1.5, 1.2];
% 
% mu_221 = [-1.2, -1.2];
% mu_222 = [-1.2, -1.5];
% 
% [Q11, ~] = qr(randn(4));
% [Q12, ~] = qr(randn(4));
% [Q21, ~] = qr(randn(4));
% [Q22, ~] = qr(randn(4));
% 
% sig11 = [1,1.5,1.5,2];
% 
% sig12 = [1,1.5,1.5,2];
% 
% sig21 = [1,1.5,2,2];
% 
% sig22 = [1,1.5,1.5,2];
% 
% 
% Sigma11 = Q11*diag(sig11)*Q11';
% Sigma12 = Q12*diag(sig12)*Q12';
% 
% Sigma21 = Q21*diag(sig21)*Q21';
% Sigma22 = Q21*diag(sig22)*Q22';
% 
% N1 = 200;
% N2 = 200;
% NTst1 = 1000;
% NTst2 = 1000;
% 
% % generate two-class data from two views; 
% X_1 = mvnrnd([mu_111, mu_211], Sigma11, N1+NTst1); % N x d matrix
% X_2 = mvnrnd([mu_121, mu_221], Sigma21, N2+NTst2);
% 
% 
% 
% y_1 = ones(N1+NTst1,1);
% y_2 = -ones(N2+NTst2,1);
% 
% X = [X_1; X_2];
% y=  [y_1; y_2];
% ind_perm = randperm(N1+NTst1+N2+NTst2);  %random permutation
% X(ind_perm,:) =X;
% y(ind_perm)= y;
% 
% ind_trn_c1 = ind_perm(1:N1);
% ind_tst_c1 = ind_perm(N1+1:N1+NTst1);
% ind_trn_c2 = ind_perm(N1+NTst1+1:N1+NTst1+N2);
% ind_tst_c2 = ind_perm(N1+NTst1+N2+1:N1+NTst1+N2+NTst2);
% 
% r_a = 0.995;
% U1 = ceil(N1*r_a); %size of unlabeled data
% L1 = N1 - U1;      %size of labeled data
% U2 = ceil(N2*r_a); %size of unlabeled data
% L2 = N2 - U2;      %size of labeled data
% 
% ind_U1 = randsample(ind_trn_c1, U1);
% ind_L1 = setdiff(ind_trn_c1, ind_U1);
% ind_U2 = randsample(ind_trn_c2, U2);
% ind_L2 = setdiff(ind_trn_c2, ind_U2);
% 
% X_U(:,:,1) = X(union(ind_U1, ind_U2),1:2);
% X_U(:,:,2) = X(union(ind_U1, ind_U2),3:4);
% y_U = y(union(ind_U1, ind_U2));
% 
% X_L(:,:,1) = X(union(ind_L1, ind_L2),1:2);
% X_L(:,:,2) = X(union(ind_L1, ind_L2),3:4);
% y_L = y(union(ind_L1, ind_L2));
% 
% 
% X_tst(:,:,1) = X(union(ind_tst_c1, ind_tst_c2),1:2);
% X_tst(:,:,2) = X(union(ind_tst_c1, ind_tst_c2),3:4);
% y_tst = y(union(ind_tst_c1, ind_tst_c2));

%load('data_exp3_09_11_14.mat')
load('data_exp_2014_09_11.mat')
%%
nU = U1+U2;
nL = L1+L2;
nTst = NTst1+NTst2;

nV = 2;
maxIter = 1;
sigma2 = 1;
epsilon = 1e-5;

%model = iniparam.inimodel;
% train two view independent classifiers y_L
model{1} = svmtrain(y_L, X_L(:,:,1)); %train is Nxd, so transpose


w11 = (model{1}.sv_coef' * full(model{1}.SVs))';
w01 = -model{1}.rho ;
 w_int1 = [w11; w01];
 
 [pred_label_v1, accuracy_v1, decision_values_v1] = ...
    svmpredict(y_tst, X_tst(:,:,1), model{1}); 
 
%model{1}.sv_coef = model{1}.sv_coef;

model{2} = svmtrain(y_L, X_L(:,:,2)); %train is Nxd, so transpose



w12 = (model{2}.sv_coef' * full(model{2}.SVs))';
w02 = -model{2}.rho ;
 w_int2 = [w12; w02];

[pred_label_v2, accuracy_v2, decision_values_v2] = ...
    svmpredict(y_tst, X_tst(:,:,2), model{2});  
 
%model{2}.sv_coef = -model{2}.sv_coef;
 
 
 
 
 
 
 


options.Kernel = 'linear';
options.KernelParam =[];

Traindata.X_U = X_U ;
Traindata.y_U = y_U ;
Traindata.X_L = X_L ;
Traindata.y_L = y_L ;
Traindata.nU = nU;
Traindata.nL = nL;
Traindata.nV = nV;
Traindata.d = 2;

Testdata.X_Tst = X_tst;
Testdata.y_Tst = y_tst;
Testdata.nTst  = nTst;
Testdata.d = 2;


param.kernelMethod = 'linear';  
param.kernelParm = 0;
param.maxIterOut = 1;
param.threOut = 1e-3;
param.maxIterMAP = 2000;
param.threMAP = 1e-4;
param.sigmaPri = 1;

 iniparam.inimodel = model;
% 
% [accuracy1, errorlist, dev_tst, prob_tst, dev_trn, prob_trn,...
%     history, history_tst] = mvmedbin(Traindata, Testdata, param, iniparam); 



param.maxIterOut = 20;

[accuracy, errorlist, dev_tst, prob_tst, dev_trn, prob_trn, wjoint...
           history, history_tst] = mvmedbin_w(Traindata, Testdata, param, iniparam);

% [accuracy2, errorlist, dev_tst, prob_tst, dev_trn, prob_trn,...
%     history, history_tst] = mvmedbin(Traindata, Testdata, param, iniparam); 
display(sprintf('accuracy v1: %f %% \t accuracy v2: %f %%', accuracy_v1(1),accuracy_v2(1)));
% display(sprintf('accuracy(iter 1): %f %%',accuracy1*100));

display(sprintf('accuracy(iter 2): %f %%',accuracy*100));




 figure(1);
subplot(1,2,1)
plot(X(ind_trn_c1,1),X(ind_trn_c1,2), 'xb' );
hold on;
plot(X(ind_trn_c2,1),X(ind_trn_c2,2), 'or' );
plot(X(ind_L1,1), X(ind_L1,2), 'xk','Linewidth',5);
plot(X(ind_L2,1), X(ind_L2,2), 'og','Linewidth',5);

plot_x1 = linspace(min([X(ind_trn_c1,1);X(ind_trn_c2,1)]), max([X(ind_trn_c1,1); X(ind_trn_c2,1)]),20);
plot(plot_x1, (w01 - w11(1)*plot_x1)./w11(2), '-.b', 'Linewidth',2 );

plot(plot_x1, ( - wjoint(1,1)*plot_x1)./wjoint(2,1), '-*b', 'Linewidth',2 );
hold off
legend('class +1','class -1', 'labeled +1', 'labeled -1', 'single view 1', 'consensus view')
title('view 1')
xlabel '1st dim'
ylabel '2nd dim'



subplot(1,2,2)
plot(X(ind_trn_c1,3),X(ind_trn_c1,4), 'xb' );
hold on;
plot(X(ind_trn_c2,3),X(ind_trn_c2,4), 'or' );
plot(X(ind_L1,3), X(ind_L1,4), 'xk','Linewidth',5);
plot(X(ind_L2,3), X(ind_L2,4), 'og','Linewidth',5);

plot_x2 = linspace(min([X(ind_trn_c1,3); X(ind_trn_c2,3)]), max([X(ind_trn_c1,3); X(ind_trn_c2,3)]), 20);

plot(plot_x2, (w02 - w12(1)*plot_x2)./w12(2), '-.b', 'Linewidth',2 );

plot(plot_x2, ( - wjoint(1,2)*plot_x2)./wjoint(2,2), '-.*b', 'Linewidth',2 );
hold off
legend('class +1','class -1','labeled +1', 'labeled -1', 'single view 2', 'consensus view')
title('view 2')
xlabel '1st dim'
ylabel '2nd dim'



