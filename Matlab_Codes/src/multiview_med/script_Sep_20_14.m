%% test on nonlinear multiview data set

addpath('./libsvm-3.18/matlab/')
addpath('../../../../../../MATLAB/cvx/');

clearvars
close all
clc

d = 2;

r_a = [0.98; 0.9; 0.7; 0.5; 0.3];

TotalR = length(r_a);
TotalSet = 1; %5;
TotalRep = 20;
accuracy_v1_array = zeros( TotalR,TotalRep,TotalSet);
accuracy_v2_array = zeros( TotalR,TotalRep,TotalSet);
accuracy_mv_array = zeros( TotalR,TotalRep,TotalSet);


nV = 2;

 [Q1, ~] = qr(randn(d));  % orthogonal transfomation to generate Sigma
 [Q2, ~] = qr(randn(d));
 sig21 = [1,1]; 
 sig22 = [1,1];
 Sigma21 = Q1*diag(sig21)*Q1';
 Sigma22 = Q2*diag(sig22)*Q2';
 
 iset = 1;
for iset = 1:TotalSet
 dist_class = -3; % -4.5+ 3*rand(1); % the distance btw two moons
 mu_21 = [1, 1]; %[linspace(2, 0.5, TotalSet)',  linspace(2, 0.5, TotalSet)'] ;
 mu_22 = -mu_21;
%% multiview sample generation
for ir =1:TotalR
    
    L1 = 5;
    L2 = L1; 
    U1 = ceil(r_a(ir)/(1-r_a(ir))*L1);
    U2 = U1; 

    N1 = L1+ U1;
    N2 = N1; 

    NTst1 = 500;
    NTst2 = NTst1; 

    nU = U1+U2;
    nL = L1+L2;
    nTst = NTst1+NTst2;

    
    
    % declare memory
    X = [];
    data = [];
    y = [];
    X_U = [];
    X_L = [];
    y_U = [];
    y_L = [];
    X_tst = [];
    y_tst = [];
    
    ind_perm = [];
    ind_c1 = [];    
    ind_c2 = [];
    ind_trn_c1 = [];
    ind_trn_c2 = [];
    ind_tst_c1 = [];
    ind_tst_c2 = [];
    ind_U1 = [];
    ind_U2 = [];
    ind_L1 = [];
    ind_L2 = [];
    
    Traindata = struct('X_U', [],'y_U', [], 'X_L', [], 'y_L', [],  ...
         'nU', 0, 'nL', 0, 'nV', nV, 'd', 2);

    Testdata = struct('X_Tst', [], 'y_Tst', [], 'nTest', 0, 'd', 2);

    
 for irep = 1:TotalRep
 %% data generation    
     X = zeros(N1+NTst1+N2+NTst2, d, nV);
     X_U = zeros(U1+U2, d, nV);
     X_L = zeros(L1+L2, d, nV);
     X_tst = zeros(NTst1+NTst2, d, nV);
     
     % view 1 is two moon set
     display('Generate two moon set...')
     data = dbmoon(N1+NTst1,dist_class,4.25,3); 
     X(:,:,1) = data(:,1:2);
     X(:,:,1) = bsxfun(@minus, X(:,:,1), mean(X(:,:,1))); % mean 0 
     
     y = data(:,3);
     ind_perm = randperm(length(y));
     X(ind_perm,:,1) = X(:,:,1);
     y(ind_perm) = y;

     ind_c1 = ind_perm(1:N1+NTst1);
     ind_c2 = ind_perm(N1+NTst1+1:N1+NTst1+N2+NTst2);

    % view 2 is bivariate Normal set
    display('Generate two bivariate Normal set...')
     X(ind_c1,:,2) = mvnrnd(mu_21, Sigma21, N1+NTst1);
     X(ind_c2,:,2) = mvnrnd(mu_22, Sigma22, N2+NTst2);
 

    ind_trn_c1 = ind_perm(1:N1);
    ind_tst_c1 = ind_perm(N1+1:N1+NTst1);
    ind_trn_c2 = ind_perm(N1+NTst1+1:N1+NTst1+N2);
    ind_tst_c2 = ind_perm(N1+NTst1+N2+1:N1+NTst1+N2+NTst2);

    ind_U1 = randsample(ind_trn_c1, U1);
    ind_L1 = setdiff(ind_trn_c1, ind_U1);
    ind_U2 = randsample(ind_trn_c2, U2);
    ind_L2 = setdiff(ind_trn_c2, ind_U2);

    X_U(:,:,1) = X(union(ind_U1, ind_U2),:,1);
    X_U(:,:,2) = X(union(ind_U1, ind_U2),:,2);

    y_U = y(union(ind_U1, ind_U2));

    X_L(:,:,1) = X(union(ind_L1, ind_L2),:,1);
    X_L(:,:,2) = X(union(ind_L1, ind_L2),:,1);
    y_L = y(union(ind_L1, ind_L2));


    X_tst(:,:,1) = X(union(ind_tst_c1, ind_tst_c2),:,1);
    X_tst(:,:,2) = X(union(ind_tst_c1, ind_tst_c2),:,2);
    y_tst = y(union(ind_tst_c1, ind_tst_c2));

 %% Initial on SVM

    optionsvm = [{'-c 1 -g 0.5'},{'-c 1 -g 0.5'}];
    model = cell(1,nV);
    accuracy_v = zeros(3,nV);
    % train two view independent classifiers y_L
    for iv = 1:nV
       model{iv} = svmtrain(y_L, X_L(:,:,iv),optionsvm{iv}); 
       
    [~, accuracy_v(:,iv), ~] = svmpredict(y_tst, X_tst(:,:,iv), model{iv}); 
    end
  
    accuracy_v1_array(ir,irep,iset) = accuracy_v(1,1);
    accuracy_v2_array(ir,irep,iset) = accuracy_v(1,2);




%% parameter setting
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

    param.regParam = 1;
    param.kernelMethod = 'rbf';  
    param.kernelParm   = sqrt(1/(2*0.5^2));
    param.maxIterOut = 20;
    param.threOut = 1e-3;
    param.maxIterMAP = 50;
    param.threMAP = 1e-4;
    param.sigmaPri = 1.5;
    param.mode = 1;
    iniparam.inimodel = model;

%% Training with 

[accuracy, errorlist, dev_tst, prob_tst, dev_trn, prob_trn,...
    history, history_tst] = mvmedbinBak(Traindata, Testdata, param, iniparam); 

accuracy_mv_array(ir, irep,iset) = accuracy*100;




 end
end
display(sprintf('accuracy v1: %f %% \t accuracy v2: %f %%', mean(accuracy_v1_array(ir,:,iset)),mean(accuracy_v2_array(ir,:,iset))));
% display(sprintf('accuracy(iter 1): %f %%',accuracy1*100));
display(sprintf('accuracy(iter 2): %f %% \n',mean(accuracy_mv_array(ir,:,iset))));
display(sprintf('=======================================================\n'))
 end


 
figure(1)
errorbar(r_a*100, mean(mean(accuracy_v1_array(:,:),2))*ones(TotalR,1), std(accuracy_v1_array(:))*ones(TotalR,1),'-.b','Linewidth',1.5);
hold on;
errorbar(r_a*100, mean(mean(accuracy_v2_array(:,:),2))*ones(TotalR,1), std(accuracy_v2_array(:))*ones(TotalR,1),'-.k','Linewidth',1.5);
errorbar(r_a*100, mean(accuracy_mv_array(:,:),2), std(accuracy_mv_array(:,:),[],2),'-r', 'Linewidth',1.5);
hold off;
xlabel 'unlabel ratio (%)'
ylabel 'Accuracy(%)'
legend('v1', 'v2', 'consensus view')
axis([0,100,0,100])
grid on



 