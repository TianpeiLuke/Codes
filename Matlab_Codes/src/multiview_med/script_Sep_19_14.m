%% test on nonlinear multiview data set

addpath('./libsvm-3.18/matlab/')
addpath('../../../../../../MATLAB/cvx/');

clearvars
close all
clc

d = 2;
TotalSet = 1;
TotalRep = 20;
TotalR = 4;

accuracy_v1_array = zeros( TotalR,TotalRep,TotalSet);
accuracy_v2_array = zeros( TotalR,TotalRep,TotalSet);
accuracy_mv_array = zeros( TotalR,TotalRep,TotalSet);

r_a = [0.98; 0.9; 0.7; 0.5; 0.3];

ir = 1;

L1 = 5;
L2 = L1; %10;
U1 = ceil(r_a(ir)/(1-r_a(ir))*L1);
U2 = U1;  %ceil(r_a(ir)/(1-r_a(ir))*L2);

N1 = L1+ U1;
N2 = N1; %L2+ U2; 

NTst1 = 500;
NTst2 = NTst1; %500;

nU = U1+U2;
nL = L1+L2;
nTst = NTst1+NTst2;

nV = 2;
maxIter = 1;
sigma2 = 1;
epsilon = 1e-5;

irep = 1;
iset = 1;

%% multiview sample generation

X = zeros(N1+NTst1+N2+NTst2, d, nV);

% view 1 is two moon set
data = dbmoon(N1+NTst1,-3,4.25,3); 
X(:,:,1) = data(:,1:2);
y = data(:,3);

X(:,:,1) = bsxfun(@minus, X(:,:,1), mean(X(:,:,1))); % mean 0 

ind_perm = randperm(length(y));
X(ind_perm,:,1) = X(:,:,1);
y(ind_perm) = y;

ind_c1 = ind_perm(1:N1+NTst1);
ind_c2 = ind_perm(N1+NTst1+1:N1+NTst1+N2+NTst2);

% view 2 is bivariate Normal set
 mu_21 = [1, 1]; %[linspace(2, 0.5, TotalSet)',  linspace(2, 0.5, TotalSet)'] ;
 mu_22 = -mu_21;
 [Q1, ~] = qr(randn(d));
 [Q2, ~] = qr(randn(d));
 sig21 = [1,1]; 
 sig22 = [1,1];
 Sigma21 = Q1*diag(sig21)*Q1';
 Sigma22 = Q2*diag(sig22)*Q2';

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

Ngrid = 100;

% the region estimate point
X_region = cell(Ngrid,2);
x1plot_v1 = linspace(min(X(:,1,1))-0.5, max(X(:,1,1))+0.5, Ngrid)';
x2plot_v1 = linspace(min(X(:,2,1))-0.5, max(X(:,2,1))+0.5, Ngrid)';
[X11, X12] = meshgrid(x1plot_v1, x2plot_v1);
vals_v1 = zeros(size(X11));
for i = 1:Ngrid
   this_X = [X11(:, i), X12(:, i)];
   X_region{i,1} = this_X;
end
%X_region{i,1} = [reshape(X11,numel(X11),1), reshape(X12,numel(X12),1)];

x1plot_v2 = linspace(min(X(:,1,2))-0.5, max(X(:,1,2))+0.5, Ngrid)';
x2plot_v2 = linspace(min(X(:,2,2))-0.5, max(X(:,2,2))+0.5, Ngrid)';
[X21, X22] = meshgrid(x1plot_v2, x2plot_v2);
vals_v2 = zeros(size(X21));
for i = 1:Ngrid
   this_X2 = [X21(:, i), X22(:, i)];
   X_region{i,2} = this_X2;
end
%X_region(:,:,2) = [reshape(X21,numel(X21),1), reshape(X22,numel(X22),1)];

% optionsvm = '-c 1 -g 0.5';
% model = svmtrain(y_L, X_L,optionsvm);
% 


%% Initial on SVM

optionsvm = [{'-c 1 -g 0.5'},{'-c 1 -g 0.5'}];

%model = iniparam.inimodel;
% train two view independent classifiers y_L
model{1} = svmtrain(y_L, X_L(:,:,1),optionsvm{1}); %train is Nxd, so transpose
% w11 = (model{1}.sv_coef' * full(model{1}.SVs))';
% w01 = -model{1}.rho ;
%  w_int1 = [w11; w01];
 
[pred_label_v1, accuracy_v1, decision_values_v1] = ...
    svmpredict(y_tst, X_tst(:,:,1), model{1}); 

for i = 1:Ngrid
[pred_region_v1] = ...
    svmpredict(sign(randn(size(X_region{i,1},1),1)),  X_region{i,1}, model{1},'-q'); 
vals_v1(:,i) = pred_region_v1;
end
 
model{2} = svmtrain(y_L, X_L(:,:,2),optionsvm{2}); %train is Nxd, so transpose
% w12 = (model{2}.sv_coef' * full(model{2}.SVs))';
% w02 = -model{2}.rho ;
%  w_int2 = [w12; w02];

[pred_label_v2, accuracy_v2, decision_values_v2] = ...
    svmpredict(y_tst, X_tst(:,:,2), model{2}); 
 
for i = 1:Ngrid
[pred_region_v2] = ...
    svmpredict(sign(randn(size(X_region{i,2},1),1)),  X_region{i,2}, model{2},'-q'); 
vals_v2(:,i) = pred_region_v2;
end

accuracy_v1_array(ir,1,1) = accuracy_v1(1);
accuracy_v2_array(ir,1,1) = accuracy_v2(1);


figure(1)
subplot(1,2,1)
contourf(X11, X12, vals_v1,'LineStyle','none');
hold on

h12{1}= plot(X(ind_U1,1,1),X(ind_U1,2,1), 'xy','Linewidth', 1, 'Markersize', 4 );
h13{1}= plot(X(ind_U2,1,1),X(ind_U2,2,1), 'or','Linewidth', 1, 'Markersize', 4 );
h14{1}= plot(X(ind_L1,1,1),X(ind_L1,2,1), '+c', 'Linewidth', 2, 'Markersize', 8);
h15{1}= plot(X(ind_L2,1,1),X(ind_L2,2,1), 'ow', 'Linewidth', 2, 'Markersize', 8);
%contour(X1, X2, vals, [0 0], 'Color', 'b');
hold off;
% only show legend for last 4 plots
h= legend([h12{1}, h13{1}, h14{1}, h15{1}], {'class +1','class -1', 'labeled +1', 'labeled -1'});
set(h,'color',[0.8,0.8,0.8]) % set background color
title(sprintf('view 1: accuracy = %.1f %%', accuracy_v1(1)))
xlabel '1st dim'
ylabel '2nd dim'

subplot(1,2,2)
contourf(X21, X22, vals_v2,'LineStyle','none');
hold on
h12{2}= plot(X(ind_U1,1,2),X(ind_U1,2,2), 'xy','Linewidth', 1, 'Markersize', 4 );
h13{2}= plot(X(ind_U2,1,2),X(ind_U2,2,2), 'or','Linewidth', 1, 'Markersize', 4 );
h14{2}= plot(X(ind_L1,1,2),X(ind_L1,2,2), '+c', 'Linewidth', 2, 'Markersize', 8);
h15{2}= plot(X(ind_L2,1,2),X(ind_L2,2,2), 'ow', 'Linewidth', 2, 'Markersize', 8);
%contour(X1, X2, vals, [0 0], 'Color', 'b');
hold off;
% only show legend for last 4 plots
h= legend([h12{2}, h13{2}, h14{2}, h15{2}], {'class +1','class -1', 'labeled +1', 'labeled -1'});
set(h,'color',[0.8,0.8,0.8]) % set background color
title(sprintf('view 2: accuracy = %.1f %%', accuracy_v2(1)))
xlabel '1st dim'
ylabel '2nd dim' 

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

%% find the decision boundary
vals_p1 = zeros(size(X11));
vals_p2 = zeros(size(X21));

vals_mv1 = vals_p1;
vals_mv2 = vals_p2;

for i = 1:Ngrid
    TestRegion.X_Tst(:,:,1) = X_region{i,1};
    TestRegion.X_Tst(:,:,2) = X_region{i,2};
    TestRegion.y_Tst = sign(randn(size(X_region{i,1},1),1));
    TestRegion.nTst  = size(X_region{i,1},1);
    TestRegion.d = 2;
    param.mode = 0;
 [~, ~, dev_tstR, prob_tstR, ~, ~,...
    historyR, history_R] = mvmedbinBak(Traindata, TestRegion, param, iniparam); 
  vals_mv1(:,i) = dev_tstR(:,1);
  vals_mv2(:,i) = dev_tstR(:,2);
  vals_p1(:,i) = prob_tstR(:,1);
  vals_p2(:,i) = prob_tstR(:,2);
end


%%

figure(2)
subplot(1,2,1)
contourf(X11, X12, sign(vals_mv1),'LineStyle','none');
%contourcmap('pink', 'Colorbar', 'on');
hold on

h22{1} = plot(X(ind_U1,1,1),X(ind_U1,2,1), 'xy','Linewidth', 1, 'Markersize', 4 );
h23{1} = plot(X(ind_U2,1,1),X(ind_U2,2,1), 'or','Linewidth', 1, 'Markersize', 4 );
h24{1} = plot(X(ind_L1,1,1),X(ind_L1,2,1), '+c', 'Linewidth', 2, 'Markersize', 8);
h25{1} = plot(X(ind_L2,1,1),X(ind_L2,2,1), 'ow', 'Linewidth', 2, 'Markersize', 8);
%contour(X1, X2, vals, [0 0], 'Color', 'b');
hold off;
h= legend([h22{1}, h23{1}, h24{1}, h25{1}], {'class +1','class -1', 'labeled +1', 'labeled -1'});
set(h,'color',[0.8,0.8,0.8]);
title(sprintf('view 1: accuracy = %.1f %%', accuracy*100))
xlabel '1st dim'
ylabel '2nd dim'

subplot(1,2,2)
contourf(X21, X22, sign(vals_mv2),'LineStyle','none');
%contourcmap('pink', 'Colorbar', 'on');
hold on
h22{2} = plot(X(ind_U1,1,2),X(ind_U1,2,2), 'xy','Linewidth', 1, 'Markersize', 4 );
h23{2} = plot(X(ind_U2,1,2),X(ind_U2,2,2), 'or','Linewidth', 1, 'Markersize', 4 );
h24{2} = plot(X(ind_L1,1,2),X(ind_L1,2,2), '+c', 'Linewidth', 2, 'Markersize', 8);
h25{2} = plot(X(ind_L2,1,2),X(ind_L2,2,2), 'ow', 'Linewidth', 2, 'Markersize', 8);
%contour(X1, X2, vals, [0 0], 'Color', 'b');
hold off;
h= legend([h22{2}, h23{2}, h24{2}, h25{2}], {'class +1','class -1', 'labeled +1', 'labeled -1'});
set(h,'color',[0.8,0.8,0.8]);
title(sprintf('view 2: accuracy = %.1f %%', accuracy*100))
xlabel '1st dim'
ylabel '2nd dim'


%------------------------------------------------------------------------

figure(3)
subplot(1,2,1)
% use contour map with 8 levels
contourf(X11, X12, vals_p1,8,'LineStyle','none');
% use contour color map with colorbar set, the second argu is the interval of
% sticks in the bar
hcm = contourcmap('jet', 0.05, 'Colorbar', 'on','Location', 'vertical');
Min = min(min(vals_p1)); Max = max(max(vals_p1));
set(gca, 'CLim', [Min, Max])
set(hcm, 'XTick', [Min, Max])
set(hcm,'XTickLabel',{num2str(Min) ,num2str(Max)}) %# don't add units here...
hold on

h32{1} = plot(X(ind_U1,1,1),X(ind_U1,2,1), 'xb','Linewidth', 1.5, 'Markersize', 4 );
h33{1} = plot(X(ind_U2,1,1),X(ind_U2,2,1), 'or','Linewidth', 1.5, 'Markersize', 4 );
h34{1} = plot(X(ind_L1,1,1),X(ind_L1,2,1), '+c', 'Linewidth', 2, 'Markersize', 8);
h35{1} = plot(X(ind_L2,1,1),X(ind_L2,2,1), 'ow', 'Linewidth', 2, 'Markersize', 8);
%contour(X1, X2, vals, [0 0], 'Color', 'b');
hold off;
% only show legend for last 4 plots
h= legend([h32{1}, h33{1}, h34{1}, h35{1}], {'class +1','class -1', 'labeled +1', 'labeled -1'});
set(h,'color',[0.8,0.8,0.8]) % set background color
title(sprintf('view 1: accuracy = %.1f %%', accuracy*100))
xlabel '1st dim'
ylabel '2nd dim'

subplot(1,2,2)
contourf(X21, X22, vals_p2,8,'LineStyle','none');
hcm = contourcmap('jet',0.05, 'Colorbar', 'on','Location', 'vertical');
Min = min(min(vals_p2)); Max = max(max(vals_p2));
set(gca, 'CLim', [Min, Max])
set(hcm, 'XTick', [Min, Max])
set(hcm,'XTickLabel',{num2str(Min) ,num2str(Max)}) %# don't add units here...
hold on
h32{2} = plot(X(ind_U1,1,2),X(ind_U1,2,2), 'xb','Linewidth', 1.5, 'Markersize', 4 );
h33{2} = plot(X(ind_U2,1,2),X(ind_U2,2,2), 'or','Linewidth', 1.5, 'Markersize', 4 );
h34{2} = plot(X(ind_L1,1,2),X(ind_L1,2,2), '+c', 'Linewidth', 2, 'Markersize', 8);
h35{2} = plot(X(ind_L2,1,2),X(ind_L2,2,2), 'ow', 'Linewidth', 2, 'Markersize', 8);
%contour(X1, X2, vals, [0 0], 'Color', 'b');
hold off;
h= legend([h32{2}, h33{2}, h34{2}, h35{2}], {'class +1','class -1', 'labeled +1', 'labeled -1'});
set(h,'color',[0.8,0.8,0.8]);
title(sprintf('view 2: accuracy = %.1f %%', accuracy*100))
xlabel '1st dim'
ylabel '2nd dim'





 