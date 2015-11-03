%%  Test on multi-view data
% written in Oct 16th 2013
clc
clear all
close all
addpath('../../../../../../MATLAB/cvx/');


% generate two view via bivariate Normal distribution

m1 = [2,-2, -3, -3]';%[0.5, 0, -3, -3]'; 
m2 =  [-2,2, 3, 3]'; %[-0.5,0, 3, 3]';

theta = pi/3;
R= zeros(4);
R(1:2,1:2) = [cos(theta), sin(theta);-sin(theta), cos(theta)];
R(3:4,3:4) = [cos(pi/2-theta), sin(pi/2-theta);-sin(pi/2-theta), cos(pi/2-theta)];

Sigma = R'*diag([2.5,1,1, 2])*R;

N1 = 200;
N2 = 200;
Ntrain1 = 3;
Ntrain2 = 3;

X1 = mvnrnd(repmat(m1',N1,1), Sigma);
y1 = ones(N1,1);
X2 = mvnrnd(repmat(m2',N2,1), Sigma);
y2 = -ones(N2,1);

% figure(1);
% plot(X1(:,1), X1(:,2), 'or','Linewidth',2);
% hold on;
% plot(X2(:,1), X2(:,2), 'xb','Linewidth',2);
% grid on;
% hold off

X= [X1; X2];
y= [y1; y2];
ind_perm = randperm(N1+N2);
ind_c1 = ind_perm(1:N1);
ind_c2 = ind_perm(N1+1:N1+N2);
X(ind_perm,:) = X; 
y(ind_perm) = y;
ind_c1_tr = ind_c1(randsample(length(ind_c1),Ntrain1));
ind_c1_utr = setdiff(ind_c1,ind_c1_tr);
ind_c2_tr = ind_c2(randsample(length(ind_c2),Ntrain2));
ind_c2_utr = setdiff(ind_c2,ind_c2_tr);

X_train1 = X(ind_c1_tr,:);
y_train1 = y(ind_c1_tr);
X_utrain1 = X(ind_c1_utr,:);
y_utrain1 = y(ind_c1_utr);

X_train2 = X(ind_c2_tr,:);
y_train2 = y(ind_c2_tr);
X_utrain2 = X(ind_c2_utr,:);
y_utrain2 = y(ind_c2_utr);

X_train = [X_train1;X_train2];
y_train = [y_train1;y_train2];


X_utrain = [X_utrain1;X_utrain2];
y_utrain = [y_utrain1;y_utrain2];



%% 

X_train_v1 = [X_train1(:,1:2);X_train2(:,1:2)];
X_utrain_v1 = [X_utrain1(:,1:2);X_utrain2(:,1:2)];
X_train_v2 = [X_train1(:,3:4);X_train2(:,3:4)];
X_utrain_v2 = [X_utrain1(:,3:4);X_utrain2(:,3:4)];

alpha_v1 = zeros(Ntrain1+Ntrain2, 1);
alpha_v2 = zeros(Ntrain1+Ntrain2, 1);
mup1        = zeros(N1+N2 -Ntrain1-Ntrain2,1);
mum1        = zeros(N1+N2 -Ntrain1-Ntrain2,1);
options.Kernel = 'linear';
options.KernelParam = 1;

%% one single joint view
K = calckernel(options,X_train,X_train);


Ntr = Ntrain1+Ntrain2;
cvx_begin sdp quiet
   variable alpha_v1(Ntr,1) nonnegative
   maximize (-0.5*alpha_v1'*(K.*(y_train*y_train'))*alpha_v1 + ones(Ntr,1)'*alpha_v1) 
   subject to
     alpha_v1 <= ones(Ntr,1);
cvx_end

Kt = calckernel(options,X_utrain,X_train);
alpha_v1(find(abs(alpha_v1)<1e-8)) = zeros(length(find(abs(alpha_v1)<1e-8)),1);

w  = ((alpha_v1.*y_train)'*X_train)';
y_pred = sign(Kt'*alpha_v1);

%% Two joint view
K2l = calckernel(options,X_train_v1,X_train_v1);
K2lu = calckernel(options,X_utrain_v1,X_train_v1);
K2u = calckernel(options,X_utrain_v1,X_utrain_v1);

K3l = calckernel(options,X_train_v2,X_train_v2);
K3lu = calckernel(options,X_utrain_v2,X_train_v2);
K3u = calckernel(options,X_utrain_v2,X_utrain_v2);


Ntr = Ntrain1+Ntrain2;
Nutr = N1+N2 -Ntrain1-Ntrain2; 

T = 20;
pre_alpha_v1 = alpha_v1;
pre_alpha_v2 = alpha_v2;
pre_mup1  = mup1;
pre_mum1  = mum1;

for t=1:T
    
   display(sprintf('iteration: %d',t));
   s = mup1 - mum1;
   smargin_v1 = ones(Ntr,1)+ y_train.*(K2lu*s);
   smargin_v2 = ones(Ntr,1)- y_train.*(K3lu*s); 

cvx_begin sdp quiet
   variable alpha_v1(Ntr,1) nonnegative
   maximize (-0.5*alpha_v1'*(K2l.*(y_train*y_train'))*alpha_v1 + ones(Ntr,1)'*alpha_v1) 
   subject to
     alpha_v1 <= ones(Ntr,1);
cvx_end
alpha_v1(find(abs(alpha_v1)<1e-8)) = zeros(length(find(abs(alpha_v1)<1e-8)),1);

cvx_begin sdp quiet
   variable alpha_v2(Ntr,1) nonnegative
   maximize (-0.5*alpha_v2'*(K3l.*(y_train*y_train'))*alpha_v2 + ones(Ntr,1)'*alpha_v2) 
   subject to
     alpha_v2 <= ones(Ntr,1);
cvx_end
alpha_v2(find(abs(alpha_v2)<1e-8)) = zeros(length(find(abs(alpha_v2)<1e-8)),1);

diff_dev = K2lu'*(alpha_v1.*y_train) - K3lu'*(alpha_v2.*y_train);

cvx_begin sdp quiet
   variable mup1(Nutr,1) nonnegative
   variable mum1(Nutr,1) nonnegative
   maximize (-0.5*(mup1-mum1)'*(K2u+K3u)*(mup1-mum1) + diff_dev'*(mup1-mum1)) 
   subject to
     mup1 <= ones(Nutr,1);
     mum1 <= ones(Nutr,1);
cvx_end
mup1(find(abs(mup1)<1e-8)) = zeros(length(find(abs(mup1)<1e-8)),1);

diff_dev_m1 = K2lu'*(alpha_v1.*y_train) - K3lu'*(alpha_v2.*y_train) - 0.5*(K2u + K3u)*mup1;

% cvx_begin sdp quiet
%    variable mum1(Nutr,1) nonnegative
%    maximize (-0.5*mum1'*(K2u+K3u)*mum1 - diff_dev_m1'*mum1) 
%    subject to
%      mum1 <= ones(Nutr,1);
% cvx_end
mum1(find(abs(mum1)<1e-8)) = zeros(length(find(abs(mum1)<1e-8)),1);

  if max(abs(alpha_v2- pre_alpha_v2)) +max(abs(alpha_v1- pre_alpha_v1)) +...
          max(abs(mup1- pre_mup1)) + max(abs(mum1- pre_mum1)) < 1e-10
      break;
  else
     pre_mup1  = mup1;
     pre_mum1  = mum1;
      pre_alpha_v2 = alpha_v2; 
      pre_alpha_v1 = alpha_v1;
  end

end
s = mup1 - mum1;

w1  = ((alpha_v1.*y_train)'*X_train_v1  -  s'*X_utrain_v1)';
w10 = ((alpha_v1.*y_train)'*X_train_v1)';
w2  = ((alpha_v2.*y_train)'*X_train_v2  +  s'*X_utrain_v2)';
w20 = ((alpha_v2.*y_train)'*X_train_v2)'; 

Ktl_v1 = calckernel(options,X_train_v1,X_utrain_v1);
Ktu_v1 = calckernel(options,X_utrain_v1,X_utrain_v1);
Ktl_v2 = calckernel(options,X_train_v2,X_utrain_v2);
Ktu_v2 = calckernel(options,X_utrain_v2,X_utrain_v2);


y_pred_v1 = sign(Ktl_v1*(alpha_v1.*y_train) - Ktu_v1*s );
y_pred_v2 = sign(Ktl_v2*(alpha_v2.*y_train) + Ktu_v2*s );
%w1  = ((alpha_v1.*y_train)'*X_train_v1)';
%y_pred_v1 = sign(Kt*alpha_v1);

mismatch_score1 = norm(y_pred_v1 - y_pred_v2,1)
mismatch_score2 = length(find(y_pred_v1 - y_pred_v2))
ind_mismatch = find(y_pred_v1 - y_pred_v2);
%%
figure(1);
set (gcf,'Position',[200,50,1200,800])
subplot(1,2,1)
plot(X(ind_c1,1), X(ind_c1,2), 'or','Linewidth',1);
hold on;
plot(X(ind_c2,1), X(ind_c2,2), 'xb','Linewidth',1);
plot(X_train1(:,1), X_train1(:,2), 'or','Linewidth',3);
plot(X_train2(:,1), X_train2(:,2), 'xb','Linewidth',3);
xLimits = get(gca,'XLim');  %# Get the range of the x axis
yLimits = get(gca,'YLim'); 
plot(X_utrain(:,1)', ( -w1(1)*X_utrain(:,1)')./w1(2), 'b', 'Linewidth',2 );
plot(X_utrain(:,1)', ( -w(1)*X_utrain(:,1)')./w(2), 'g', 'Linewidth',2 );
%plot(X_utrain(:,1)', ( -w10(1)*X_utrain(:,1)')./w10(2), '-.m', 'Linewidth',1 );
grid on;
axis([xLimits,yLimits]);
hold off
legend('c1-unlabeled','c2-unlabeled','c1-labeled','c2-labeled','multi-view', 'single-view')
subplot(1,2,2)
plot(X(ind_c1,3), X(ind_c1,4), 'or','Linewidth',1);
hold on;
plot(X(ind_c2,3), X(ind_c2,4), 'xb','Linewidth',1);
plot(X_train1(:,3), X_train1(:,4), 'or','Linewidth',3);
plot(X_train2(:,3), X_train2(:,4), 'xb','Linewidth',3);
xLimits = get(gca,'XLim');  %# Get the range of the x axis
yLimits = get(gca,'YLim');
plot(X_utrain(:,3)', ( -w2(1)*X_utrain(:,3)')./w2(2), 'b', 'Linewidth',2 );
plot(X_utrain(:,3)', ( -w(3)*X_utrain(:,3)')./w(4), 'g', 'Linewidth',2 );
%plot(X_utrain(:,3)', ( -w20(1)*X_utrain(:,3)')./w20(2), '-.m', 'Linewidth',1 );
grid on;
axis([xLimits,yLimits]);
hold off
legend('c1-unlabeled','c2-unlabeled','c1-labeled','c2-labeled','multi-view', 'single-view')

figure(2)
plot(s)


