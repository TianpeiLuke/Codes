%% Script to test mvmedbin.m 
addpath('./libsvm-3.18/matlab/')
addpath('../../../../../../MATLAB/cvx/');
%clear all
clearvars
close all
clc

d = 4;
TotalSet = 5;
TotalRep = 20;
TotalR = 4;

accuracy_v1_array = zeros( TotalR,TotalRep,TotalSet);
accuracy_v2_array = zeros( TotalR,TotalRep,TotalSet);
accuracy_mv_array = zeros( TotalR,TotalRep,TotalSet);
 % U/(L+U)
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

load('data_exp_2014_09_11.mat')
clear X_1 X_2 y_1 y_2 X y ind_trn_c1 ind_tst_c1 ind_trn_c2 ind_tst_c2
clear X_U y_U X_L y_L X_tst y_tst
clear r_a
r_a = [0.9; 0.7; 0.5; 0.3];
irep = 1;


mu_111 = [linspace(-2, -0.5, TotalSet)', - linspace(-2, -0.5, TotalSet)'] ;
mu_121 = -mu_111;
mu_211 = [linspace(2, 0.5, TotalSet)',  linspace(2, 0.5, TotalSet)'] ;
mu_221 = -mu_211;
KLdiv = zeros(1,TotalSet);

for iset = 1:TotalSet


KLdiv21 = 0.5*(log(det(Sigma21)/det(Sigma11)) - d + trace(Sigma21\Sigma11)...
    + ([mu_121(iset,:), mu_221(iset,:)]-[mu_111(iset,:), mu_211(iset,:)])*...
    (Sigma21\([mu_121(iset,:), mu_221(iset,:)]-[mu_111(iset,:), mu_211(iset,:)])'));
   
KLdiv12 = 0.5*(log(det(Sigma11)/det(Sigma21)) - d + trace(Sigma11\Sigma21)...
    + ([mu_111(iset,:), mu_211(iset,:)]-[mu_121(iset,:), mu_221(iset,:)])*...
    (Sigma11\([mu_111(iset,:), mu_211(iset,:)]-[mu_121(iset,:), mu_221(iset,:)])'));
    
KLdiv(iset) = 0.5*KLdiv21 + 0.5*KLdiv12;

for ir =1:TotalR
    irep = 0;
 while irep < TotalRep
 irep = irep+1;
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

% 
% N1 = 200;
% N2 = 200;
% NTst1 = 1000;
% NTst2 = 1000;
%  

L1 = 20;
L2 = 20;
U1 = ceil(r_a(ir)/(1-r_a(ir))*L1);
U2 = ceil(r_a(ir)/(1-r_a(ir))*L2);

N1 = L1+ U1;
N2 = L2+ U2; 

% % generate two-class data from two views; 
X_1 = mvnrnd([mu_111(iset,:), mu_211(iset,:)], Sigma11, N1+NTst1); % N x d matrix
X_2 = mvnrnd([mu_121(iset,:), mu_221(iset,:)], Sigma21, N2+NTst2);



y_1 = ones(N1+NTst1,1);
y_2 = -ones(N2+NTst2,1);

X = [X_1; X_2];
y=  [y_1; y_2];
ind_perm = randperm(N1+NTst1+N2+NTst2);  %random permutation
X(ind_perm,:) =X;
y(ind_perm)= y;

ind_trn_c1 = ind_perm(1:N1);
ind_tst_c1 = ind_perm(N1+1:N1+NTst1);
ind_trn_c2 = ind_perm(N1+NTst1+1:N1+NTst1+N2);
ind_tst_c2 = ind_perm(N1+NTst1+N2+1:N1+NTst1+N2+NTst2);


% U1 = ceil(N1*r_a(ir)); %size of unlabeled data
% L1 = N1 - U1;      %size of labeled data
% U2 = ceil(N2*r_a(ir)); %size of unlabeled data
% L2 = N2 - U2;      %size of labeled data

ind_U1 = randsample(ind_trn_c1, U1);
ind_L1 = setdiff(ind_trn_c1, ind_U1);
ind_U2 = randsample(ind_trn_c2, U2);
ind_L2 = setdiff(ind_trn_c2, ind_U2);

X_U(:,:,1) = X(union(ind_U1, ind_U2),1:2);
X_U(:,:,2) = X(union(ind_U1, ind_U2),3:4);
y_U = y(union(ind_U1, ind_U2));

X_L(:,:,1) = X(union(ind_L1, ind_L2),1:2);
X_L(:,:,2) = X(union(ind_L1, ind_L2),3:4);
y_L = y(union(ind_L1, ind_L2));


X_tst(:,:,1) = X(union(ind_tst_c1, ind_tst_c2),1:2);
X_tst(:,:,2) = X(union(ind_tst_c1, ind_tst_c2),3:4);
y_tst = y(union(ind_tst_c1, ind_tst_c2));

%load('data_exp3_09_11_14.mat')

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
 
 
model{2} = svmtrain(y_L, X_L(:,:,2)); %train is Nxd, so transpose
w12 = (model{2}.sv_coef' * full(model{2}.SVs))';
w02 = -model{2}.rho ;
 w_int2 = [w12; w02];

 [pred_label_v2, accuracy_v2, decision_values_v2] = ...
    svmpredict(y_tst, X_tst(:,:,2), model{2}); 
 
accuracy_v1_array(ir,irep,iset) = accuracy_v1(1);
accuracy_v2_array(ir,irep,iset) = accuracy_v2(1);
 
%  figure(1);
% subplot(1,2,1)
% plot(X(ind_trn_c1,1),X(ind_trn_c1,2), 'xb' );
% hold on;
% plot(X(ind_trn_c2,1),X(ind_trn_c2,2), 'or' );
% 
% plot_x1 = linspace(min([X(ind_trn_c1,1);X(ind_trn_c2,1)]), max([X(ind_trn_c1,1); X(ind_trn_c2,1)]));
% plot(plot_x1, (-w01 - w11(1)*plot_x1)./w11(2), '-.b', 'Linewidth',1.5 );
% hold off
% title('view 1')
% 
% subplot(1,2,2)
% plot(X(ind_trn_c1,3),X(ind_trn_c1,4), 'xb' );
% hold on;
% plot(X(ind_trn_c2,3),X(ind_trn_c2,4), 'or' );
% 
% plot_x2 = linspace(min([X(ind_trn_c1,3); X(ind_trn_c2,3)]), max([X(ind_trn_c1,3); X(ind_trn_c2,3)]));
% plot(plot_x2, (-w02 - w12(1)*plot_x2)./w12(2), '-.b', 'Linewidth',1.5 );
% hold off
% title('view 2')
 
 
 
 
 
 
 


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

param.regParam = 2;
param.kernelMethod = 'linear';  
param.kernelParm = 0;
param.maxIterOut = 20;
param.threOut = 1e-3;
param.maxIterMAP = 50;
param.threMAP = 1e-4;
param.sigmaPri = 0.01;
param.mode = 1;
iniparam.inimodel = model;

% [accuracy1, errorlist, dev_tst, prob_tst, dev_trn, prob_trn,...
%     history, history_tst] = mvmedbin(Traindata, Testdata, param, iniparam); 



%param.maxIterOut = 20;

[accuracy, errorlist, dev_tst, prob_tst, dev_trn, prob_trn,...
    history, history_tst,programflag] = mvmedbinBak(Traindata, Testdata, param, iniparam); 


if programflag == 0
    irep = irep-1;
    continue;
end
accuracy_mv_array(ir, irep,iset) = accuracy*100;


 end
  X_1 = [];
  X_2 = [];
  X_U = [];
  X_L = [];
  y_1 = [];
  y_2 = [];
  y_U = [];
  y = [];
  y_L= [];
  ind_perm = [];
ind_trn_c1 =[];
ind_tst_c1 = [];
ind_trn_c2 = [];
ind_tst_c2 = [];
U1 = 0; %size of unlabeled data
L1 =0;      %size of labeled data
U2 = 0; %size of unlabeled data
L2 = 0;      %size of labeled data

ind_U1 =[];
ind_L1 =[];
ind_U2 = [];
ind_L2 = [];
end
display(sprintf('accuracy v1: %f %% \t accuracy v2: %f %%', mean(accuracy_v1_array(ir,:,iset)),mean(accuracy_v2_array(ir,:,iset))));
% display(sprintf('accuracy(iter 1): %f %%',accuracy1*100));
display(sprintf('accuracy(iter 2): %f %%',mean(accuracy_mv_array(ir,:,iset))));
end

figure(2)
color = [{'-r'},{'-b'},{'-g'},{'-.b'},{'-.g'}];
legendstr = [{sprintf('symmetric KL divergence = %.f',KLdiv(TotalSet))}];
errorbar(r_a*100, mean(accuracy_mv_array(:,:,TotalSet),2), std(accuracy_mv_array(:,:,TotalSet),[],2),color{TotalSet},'Linewidth',1.5);
hold on;

for iset=TotalSet-1:-1:1
errorbar(r_a*100, mean(accuracy_mv_array(:,:,iset),2), std(accuracy_mv_array(:,:,iset),[],2),color{iset},'Linewidth',1.5);
legendstr = [legendstr, {sprintf('symmetric KL divergence = %.f',KLdiv(iset))}];
end
hold off;
xlabel 'unlabel ratio (%)'
ylabel 'Accuracy(%)'
legend(legendstr)
axis([0,100,0,100])
grid on

