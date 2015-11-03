%% Test on the two-view unnormalized data
%addpath('./libsvm-weights/matlab')
addpath('./libsvm-3.16/matlab/')
addpath('../../../../../../MATLAB/cvx/');
clear all
close all
clc


mu_111 = [-1.1, 1];
mu_112 = [-1.1, 1];

mu_121 = [1, -1.1];
mu_122 = [1, -1.1];

mu_211 = [1.2, 1.2];
mu_212 = [1.5, 1.2];

mu_221 = [-1.2, -1.2];
mu_222 = [-1.2, -1.5];

[Q11, ~] = qr(randn(4));
[Q12, ~] = qr(randn(4));
[Q21, ~] = qr(randn(4));
[Q22, ~] = qr(randn(4));

sig11 = [1,1.5,1.5,2];

sig12 = [1,1.5,1.5,2];

sig21 = [1,1.5,2,2];

sig22 = [1,1.5,1.5,2];


Sigma11 = Q11*diag(sig11)*Q11';
Sigma12 = Q12*diag(sig12)*Q12';

Sigma21 = Q21*diag(sig21)*Q21';
Sigma22 = Q21*diag(sig22)*Q22';

N1 = 200;
N2 = 200;

% generate two-class data from two views; 
X_1 = mvnrnd([mu_111, mu_211], Sigma11, N1)'; % dx N matrix
X_2 = mvnrnd([mu_121, mu_221], Sigma21, N2)';


y_1 = ones(N1,1);
y_2 = -ones(N2,1);

X = [X_1, X_2];
y=  [y_1; y_2];
ind_perm = randperm(N1+N2);  %random permutation
X(:,ind_perm) =X;
y(ind_perm)= y;

ind_c1 = ind_perm(1:N1);
ind_c2 = ind_perm(N1+1:N1+N2);


r_a = 0.995;
U1 = ceil(N1*r_a); %size of unlabeled data
L1 = N1 - U1;      %size of labeled data
U2 = ceil(N2*r_a); %size of unlabeled data
L2 = N2 - U2;      %size of labeled data

ind_U1 = randsample(ind_c1, U1);
ind_L1 = setdiff(ind_c1, ind_U1);
ind_U2 = randsample(ind_c2, U2);
ind_L2 = setdiff(ind_c2, ind_U2);

X_U = X(:,union(ind_U1, ind_U2));
y_U = y(union(ind_U1, ind_U2));
X_L = X(:,union(ind_L1, ind_L2));
y_L = y(union(ind_L1, ind_L2));





%% parameters
str_all = [{'-c 1 -t 0 -g 1 -b 0'}, {'-c 1 -t 0 -g 1 -b 0 -h 0'}]; 
eps = 1e-5;
%w_pre = zeros(1,d+1);

%N = 100;




sigmoid = @(x)(1./(1+ exp(-x)));



%[predict_label2, accuracy2, dec_values2] = svmpredict(y_test_label, X_test, model2);
 %error1(rr,ii) = 100-accuracy2(1);
% error(rr, ii, 1) = 100-accuracy2(1);

%% initialization
q = 0.5*ones(U1+U2,1); % pseudo-label probablity

% train two view independent classifiers y_L
model1 = svmtrain(y_L, X_L(1:2,:)'); %train is Nxd, so transpose
w11 = (model1.sv_coef' * full(model1.SVs))';
w01 = -model1.rho ;
 w_int1 = [w11; w01];
 
 
model2 = svmtrain(y_L, X_L(3:4,:)'); %train is Nxd, so transpose
w12 = (model2.sv_coef' * full(model2.SVs))';
w02 = -model2.rho ;
 w_int2 = [w12; w02];
 
q= sigmoid(1/2*w_int1'*[X_U(1:2,:);ones(1,size(X_U(1:2,:),2))] + ...
      1/2*w_int2'*[X_U(3:4,:);ones(1,size(X_U(3:4,:),2))] )'; 

% verify the initial performance  
gnd_tr  = sigmoid(100*y_U);  
error01 = (sigmoid([X_U(1:2,:);ones(1,size(X_U(1:2,:),2))]'*w_int1)-gnd_tr);
error01(find(error01<1e-5)) = zeros(length(find(error01<1e-5)),1);
rate01 = length(find(abs(error01)>0))/length(error01) 

error02 = (sigmoid([X_U(3:4,:);ones(1,size(X_U(3:4,:),2))]'*w_int2)-gnd_tr);
error02(find(error02<1e-5)) = zeros(length(find(error02<1e-5)),1);
rate02 = length(find(abs(error02)>0))/length(error02) 

errorg = (q-gnd_tr);
errorg(find(errorg<1e-5)) = zeros(length(find(errorg<1e-5)),1);
rateg = sum(abs(errorg))/length(errorg)   
  
% update M
 thr = 1e-2;
 v1 = 1/2*1./(1+cosh(w_int1'*[X_U(1:2,:);ones(1,size(X_U(1:2,:),2))]));
 %v1(find(v1 < thr)) = zeros(length(find(v1 < thr)),1);
 
 M1 = diag(v1);
 
 v2 = 1/2*1./(1+cosh(w_int1'*[X_U(3:4,:);ones(1,size(X_U(3:4,:),2))]));
 %v2(find(v2 < thr)) = zeros(length(find(v2 < thr)),1);
 M2 = diag(v2);
 
%kernel matrix
K_l1=  X_L(1:2,:)'*X_L(1:2,:); %cross-kernel
K_u1=  X_U(1:2,:)'*X_U(1:2,:); %self-kernel
K_l2=  X_L(3:4,:)'*X_L(3:4,:);
K_u2=  X_U(3:4,:)'*X_U(3:4,:);
K_ul1 = X_U(1:2,:)'*X_L(1:2,:);
K_ul2 = X_U(3:4,:)'*X_L(3:4,:);

T= 2000;
flag_w1 = 1;
t = 0;
w_map1 = w11;
w_map2 = w12;
% history1 = zeros(1,T);
% historyw1 = zeros(2,T);
% historydiff = zeros(1,T);
% historydiff2 = zeros(1,T);
% loss1 = zeros(1,T);
% loss2 = zeros(1,T);

rho1 = 0.1;
rho2 = 0.1;

%% find MAP w for unlabeled part
while(flag_w1 && t<T)
t = t+1; t

% update p
p1 = sigmoid(X_U(1:2,:)'*w_map1);
p2 = sigmoid(X_U(3:4,:)'*w_map2);


% update w_map1, w_map2
w_pre1 = w_map1;
% historyw1(:,t)= w_map1;
% historydiff(t)= sum(abs(q - p1));
% historydiff2(t)= sum(abs(q - p2));

rho1 = 1;%exp(-1e-4*t);
%loss1(t) = norm(w_map1 - rho1*X_U(1:2,:)*(q-p1));

w_map1 =  X_U(1:2,:)*((pinv(M1) + K_u1)\X_U(1:2,:)'*w_map1)+ X_U(1:2,:)*(q-p1) ...
    - X_U(1:2,:)*((pinv(M1) + K_u1)\ K_u1*(q-p1) );


%history1(t) = norm(w_map1-w_pre1);
 

w_pre2 = w_map2;
%loss2(t) = norm(w_map2 - rho1*X_U(3:4,:)*(q-p2));

w_map2 =  X_U(3:4,:)*((pinv(M2) + K_u2)\X_U(3:4,:)'*w_map2)+ X_U(3:4,:)*(q-p2) ...
    - X_U(3:4,:)*((pinv(M2) + K_u2)\ K_u2*(q-p2) );

% update M
 v1 = 1/2*1./(1+cosh(w_map1'*X_U(1:2,:)));
 %v1(find(v1 < thr)) = zeros(length(find(v1 < thr)),1);
 M1 = diag(v1);
 
 v2 = 1/2*1./(1+cosh(w_map2'*X_U(3:4,:)));
 %v2(find(v2 < thr)) = zeros(length(find(v2 < thr)),1);
 M2 = diag(v2);

 if norm(w_map1-w_pre1)<1e-5 && norm(w_map2-w_pre2)<1e-5
   flag_w1 = 0;
%    history1(t+1:end) = [];
%    historydiff(t+1:end) = [];
%    historydiff2(t+1:end) = [];
%    historyw1(:,t+1:end) = [];
%    loss1(:,t+1:end) = [];
%    loss2(:,t+1:end) = [];
 end
end
rateg

error11 = (sigmoid(X_U(1:2,:)'*w_map1)- gnd_tr);
error11(find(error11<1e-5)) = zeros(length(find(error11<1e-5)),1);
rate11 = sum(abs(error11))/length(error11) 


error12 = (sigmoid(X_U(3:4,:)'*w_map2)- gnd_tr);
error12(find(error12<1e-5)) = zeros(length(find(error12<1e-5)),1);
rate12 = sum(abs(error12))/length(error12) 

%% find sparse dual variables for labeled part
 v1 = 1/2*1./(1+cosh(w_map1'*X_U(1:2,:)));
 %v1(find(v1 < thr)) = zeros(length(find(v1 < thr)),1);
 M1 = diag(v1);
 v2 = 1/2*1./(1+cosh(w_map2'*X_U(3:4,:)));
 %v2(find(v2 < thr)) = zeros(length(find(v2 < thr)),1);
 M2 = diag(v2);
 len = length(y_L);
 
b1 =  (w_map1'*X_L(1:2,:))'.*y_L;
b2 =  (w_map2'*X_L(3:4,:))'.*y_L;
eones = ones(length(y_L), 1);

P1 =  (K_l1 - (K_ul1'*((pinv(M1) + K_u1)\K_ul1))).*(y_L*y_L');
[U1, S1] = eig(P1);
%P1 =  (X_L(1:2,:)'*inv(X_U(1:2,:)*M1*X_U(1:2,:)' + eye(2))*X_L(1:2,:)).*(y_L*y_L');
P2 =  (K_l2 - (K_ul2'*((pinv(M2) + K_u2)\K_ul2))).*(y_L*y_L');
[U2, S2] = eig(P2);
%P2 =  (X_L(3:4,:)'*inv(X_U(3:4,:)*M2*X_U(3:4,:)' + eye(2))*X_L(3:4,:)).*(y_L*y_L');

 cvx_begin sdp
   cvx_precision high
   variable alpha1(len) nonnegative;
   %minimize(-(eones-b1)'*alpha1 + 0.5*alpha1'*P1*alpha1 )
   minimize(-(eones-b1)'*alpha1 + 0.5*alpha1'*(U1*S1*U1')*alpha1 )
     subject to
       alpha1 <= 1;
 cvx_end 

 cvx_begin sdp
   cvx_precision high
   variable alpha2(len) nonnegative;
   %minimize(-(eones-b2)'*alpha2 + 0.5*alpha2'*P2*alpha2 )
   minimize(-(eones-b2)'*alpha2 + 0.5*alpha2'*(U2*S2*U2')*alpha2 ) 
     subject to
       alpha2 <= 1;
 cvx_end 
%% new model
f1 = zeros(N1+N2,1);
f2 = zeros(N1+N2,1);

%unlabeled part
f1(union(ind_U1, ind_U2)) =  X_U(1:2,:)'*w_map1 + K_ul1*(y_L.*alpha1) ...
                             - K_u1*((pinv(M1) + K_u1)\K_ul1)*(y_L.*alpha1);

f2(union(ind_U1, ind_U2)) =  X_U(3:4,:)'*w_map2 +  K_ul2*(y_L.*alpha2) ...
                             - K_u2*((pinv(M2) + K_u2)\K_ul2)*(y_L.*alpha2);


%lableled part 
f1(union(ind_L1, ind_L2)) =  X_L(1:2,:)'*w_map1 + K_l1*(y_L.*alpha1) ...
                             - K_ul1'*((pinv(M1) + K_u1)\K_ul1)*(y_L.*alpha1);

f2(union(ind_L1, ind_L2)) =  X_L(3:4,:)'*w_map2 + K_l2*(y_L.*alpha2) ...
                             - K_ul2'*((pinv(M2) + K_u2)\K_ul2)*(y_L.*alpha2);


%new consensus view
q_new= sigmoid(0.5*f1(union(ind_U1, ind_U2)) ...
    + 0.5*f2(union(ind_U1, ind_U2)));

error = (q_new-gnd_tr);
error(find(error<1e-5)) = zeros(length(find(error<1e-5)),1);
rate = sum(abs(error))/length(error); 
%rateg
display(sprintf('g old %f, g new %f',rateg, rate))

pf1 = sigmoid(f1(union(ind_U1, ind_U2)));
error1 = (sigmoid(f1(union(ind_U1, ind_U2)))-gnd_tr);
error1(find(error1<1e-5)) = zeros(length(find(error1<1e-5)),1);
rate1 = sum(abs(error1))/length(error1);
%rate11
display(sprintf('v1 old %f, v1 new %f',rate01, rate1))
display(sprintf('v1 old %f, v1 map %f',rate01, rate11))

error2 = (sigmoid(f2(union(ind_U1, ind_U2)))-gnd_tr);
error2(find(error2<1e-5)) = zeros(length(find(error2<1e-5)),1);
rate2 = sum(abs(error2))/length(error2);
%rate12
display(sprintf('v2 old %f, v2 new %f',rate02, rate2))
display(sprintf('v2 old %f, v2 map %f',rate02, rate12))
%%
figure(1);
subplot(1,2,1)
plot(X(1,ind_c1),X(2,ind_c1), 'xb' );
hold on;
plot(X(1,ind_c1),X(2,ind_c1), 'xb' );
plot(X(1,ind_c2),X(2,ind_c2), 'or' );

plot_x1 = linspace(min([X(1,ind_c1), X(1,ind_c2)]), max([X(1,ind_c1), X(1,ind_c2)]));
plot(plot_x1, (-w01 - w11(1)*plot_x1)./w11(2), '-.b', 'Linewidth',1.5 );
hold off
title('view 1')

subplot(1,2,2)
plot(X(3,ind_c1),X(4,ind_c1), 'xb' );
hold on;
plot(X(3,ind_c2),X(4,ind_c2), 'or' );

plot_x2 = linspace(min([X(3,ind_c1), X(3,ind_c2)]), max([X(3,ind_c1), X(3,ind_c2)]));
plot(plot_x2, (-w02 - w12(1)*plot_x2)./w12(2), '-.b', 'Linewidth',1.5 );
hold off
title('view 2')