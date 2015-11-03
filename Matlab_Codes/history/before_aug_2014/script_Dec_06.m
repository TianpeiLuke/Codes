clc
clear all
close all
addpath('../../../../../../MATLAB/cvx/');


% generate two view via bivariate Normal distribution

m1 = [0.5,-1, -1, -1.5]';%[0.5, 0, -3, -3]'; 
m2 = [-1, 1.5, 1, 1]'; %[-0.5,0, 3, 3]';

theta = pi/3;
R= zeros(4);
R(1:2,1:2) = [cos(theta), sin(theta);-sin(theta), cos(theta)];
R(3:4,3:4) = [cos(pi/2-theta), sin(pi/2-theta);-sin(pi/2-theta), cos(pi/2-theta)];

Sigma = R'*diag([2.5,1,1, 2])*R;

N1 = 200;
N2 = 200;
Ntrain1 = 1;
Ntrain2 = 1;
Ntest1 = 1000;
Ntest2 = 1000;

run = 1;
error = zeros(2,run);

for r=1:run
X1 = mvnrnd(repmat(m1',N1,1), Sigma);
y1 = ones(N1,1);
X2 = mvnrnd(repmat(m2',N2,1), Sigma);
y2 = -ones(N2,1);

X_test1 =  mvnrnd(repmat(m1',Ntest1,1), Sigma);
y_test1 =  ones(Ntest1,1);
X_test2 =  mvnrnd(repmat(m2',Ntest2,1), Sigma);
y_test2 = -ones(Ntest2, 1);


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

X_test = [X_test1; X_test2];
y_test = [y_test1; y_test2];
ind_perm = randperm(Ntest1+Ntest2);
X_test(ind_perm,:) = X_test; 
y_test(ind_perm) = y_test;


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
max_norm_v1 = max(sqrt(sum(X_utrain(:,[1,2]).^2,2)));
max_norm_v2 = max(sqrt(sum(X_utrain(:,[3,4]).^2,2)));
%%
T = 10;
Ntr = Ntrain1+Ntrain2;
Nutr = N1+N2 -Ntrain1-Ntrain2; 

history_Kll = cell(2,T);
history_Klu = cell(2,T);
history_w = cell(2,T);
history_q = cell(1,T);
history_KL = zeros(2,T);
history_hdist = zeros(1,T);
history_phi = cell(1,T);
%history_hdist = cell(1,T);

% compute the dual variable
K_ll_1 =(y_train*y_train').*(X_train(:,[1:2])*X_train(:,[1:2])'); 
%K_lu_1 =  (X_train(:,[1:2])*X_utrain(:,[1:2])');
K_ll_2 =(y_train*y_train').*(X_train(:,[3:4])*X_train(:,[3:4])'); 
%K_lu_2 =  (X_train(:,[3:4])*X_utrain(:,[3:4])');
history_Kll{1,1} =  K_ll_1;
history_Kll{2,1} =  K_ll_2;


lin_vec_v1 =  - ones(Ntr, 1);
lin_vec_v2 =  - ones(Ntr, 1);
q = 0.5*ones(Nutr, 1);



%% compare with single view 
cvx_begin sdp 
   variable alpha_s1(Ntr,1) nonnegative
   maximize (-0.5*alpha_s1'*(K_ll_1)*(alpha_s1) + ones(Ntr,1)'*alpha_s1) 
   subject to
     alpha_s1 <= ones(Ntr,1);
cvx_end

cvx_begin sdp 
   variable alpha_s2(Ntr,1) nonnegative
   maximize (-0.5*alpha_s2'*(K_ll_2)*(alpha_s2) + ones(Ntr,1)'*alpha_s2) 
   subject to
     alpha_s2 <= ones(Ntr,1);
cvx_end

w_s_1 = (sum(diag(y_train.*alpha_s1)*X_train(:,[1:2]), 1))' ;
w_s_2 = (sum(diag(y_train.*alpha_s2)*X_train(:,[3:4]), 1))' ;

for t=1:T
%% construct the MED classifier
lin_vec_v1 =  -(y_train*(q - 0.5*ones(Nutr, 1))').*(X_train(:,[1:2])*X_utrain(:,[1:2])')*ones(Nutr,1) + ones(Ntr, 1);   
lin_vec_v2 =  -(y_train*(q - 0.5*ones(Nutr, 1))').*(X_train(:,[3:4])*X_utrain(:,[3:4])')*ones(Nutr,1) + ones(Ntr, 1);   
    
cvx_begin sdp 
   variable alpha_v1(Ntr,1) nonnegative
   maximize (-0.5*alpha_v1'*(K_ll_1)*(alpha_v1) + lin_vec_v1'*alpha_v1) 
   subject to
     alpha_v1 <= ones(Ntr,1);
cvx_end

cvx_begin sdp 
   variable alpha_v2(Ntr,1) nonnegative
   maximize (-0.5*alpha_v2'*(K_ll_2)*(alpha_v2) + lin_vec_v2'*alpha_v2) 
   subject to
     alpha_v2 <= ones(Ntr,1);
cvx_end

w_t_1 = (sum(diag(y_train.*alpha_v1)*X_train(:,[1:2]), 1))' + (sum(diag(q - 0.5*ones(Nutr, 1))*X_utrain(:,[1:2]),1))';
w_t_2 = (sum(diag(y_train.*alpha_v2)*X_train(:,[3:4]), 1))' + (sum(diag(q - 0.5*ones(Nutr, 1))*X_utrain(:,[3:4]),1))';

history_w{1,t} = w_t_1;
history_w{2,t} = w_t_2;

%% step 2: allocate the pseudo-label
q = 1./(1 + exp(-0.5*X_utrain(:,[1:2])*w_t_1- 0.5*X_utrain(:,[3:4])*w_t_2));
history_q{t} = q; 

%% 
p1 = 1./(1 + exp(-X_utrain(:,[1:2])*w_t_1));
p2 = 1./(1 + exp(-X_utrain(:,[3:4])*w_t_2));
p  = 0.5*(y_utrain+ ones(Nutr,1));

KL_1 = sum(filt(q.*log(q./p1)) + filt((ones(Nutr,1)-q).*log((ones(Nutr,1)-q)./(ones(Nutr,1)-p1))));
KL_2 = sum(filt(q.*log(q./p2)) + filt((ones(Nutr,1)-q).*log((ones(Nutr,1)-q)./(ones(Nutr,1)-p2))));
%h_dist = 1/sqrt(2)*sqrt((sqrt(p1) - sqrt(p2)).^2 + (sqrt(ones(Nutr,1)-p1) - sqrt(ones(Nutr,1)-p2)).^2);
%b_dist = -log(sqrt(p1.*p2) + sqrt((ones(Nutr,1)-p1).*(ones(Nutr,1)-p2)));
h_dist = 1/sqrt(2)*sum(sqrt((sqrt(q) - sqrt(p)).^2 + (sqrt(ones(Nutr,1)-q) - sqrt(ones(Nutr,1)-p)).^2))/Nutr;


history_hdist(t) = h_dist;
history_KL(1,t)=KL_1; 
history_KL(2,t)=KL_2; 
%%
phi_v1 =  0.5*(q - ones(Nutr, 1)).*...
        (-X_utrain(:,[1:2])*w_t_1./max_norm_v1 + X_utrain(:,[3:4])*w_t_2./max_norm_v2)...
        - log(1 + exp(-0.5*X_utrain(:,[1:2])*w_t_1./max_norm_v1- 0.5*X_utrain(:,[3:4])*w_t_2./max_norm_v2))...
       +  log(1 + exp(- X_utrain(:,[1:2])*w_t_1./max_norm_v1 )) ;

phi_v2 =  0.5*(q - ones(Nutr, 1)).*...
        (X_utrain(:,[1:2])*w_t_1./max_norm_v1 - X_utrain(:,[3:4])*w_t_2./max_norm_v2)...
        - log(1 + exp(-0.5*X_utrain(:,[1:2])*w_t_1./max_norm_v1- 0.5*X_utrain(:,[3:4])*w_t_2./max_norm_v2))...
       +  log(1 + exp(- X_utrain(:,[3:4])*w_t_2./max_norm_v2 )) ;   
   
   
history_phi{t} = min([phi_v1,phi_v2],[],2);

end
%%

y_pred1 = sign(X_test(:,[1:2])*w_t_1);
y_pred2 = sign(X_test(:,[3:4])*w_t_2);

error_v1 = length(find(y_test~=y_pred1))/length(y_test);
error_v2 = length(find(y_test~=y_pred2))/length(y_test);

error(1,r) = error_v1;
error(2,r) = error_v2;
end

error_mean_1 = mean(error(1,:));
error_std_1 = std(error(1,:));
error_mean_2 = mean(error(2,:));
error_std_2 = std(error(2,:));
%%
figure(1);
set (gcf,'Position',[200,50,1200,800])
subplot(1,2,1)
plot(X(ind_c1,1), X(ind_c1,2), 'or','Linewidth',1);
hold on;
plot(X(ind_c2,1), X(ind_c2,2), 'xb','Linewidth',1);
plot(X_train1(:,1), X_train1(:,2), 'og','Linewidth',3);
plot(X_train2(:,1), X_train2(:,2), 'xc','Linewidth',3);
xLimits = get(gca,'XLim');  %# Get the range of the x axis
yLimits = get(gca,'YLim');

plot(X_utrain(:,1)', ( -w_t_1(1)*X_utrain(:,1)')./w_t_1(2), 'b', 'Linewidth',2 );
plot(X_utrain(:,1)', ( -w_s_1(1)*X_utrain(:,1)')./w_s_1(2), '--m', 'Linewidth',1.5 );
%plot(X_utrain(:,1)', ( -w(1)*X_utrain(:,1)')./w(2), 'g', 'Linewidth',2 );
%plot(X_utrain(:,1)', ( -w10(1)*X_utrain(:,1)')./w10(2), '-.m', 'Linewidth',1 );
grid on;
axis([xLimits,yLimits]);
hold off
legend('c1-unlabeled','c2-unlabeled','c1-labeled','c2-labeled','multi-view', 'single-view')
title('view 1')
subplot(1,2,2)
plot(X(ind_c1,3), X(ind_c1,4), 'or','Linewidth',1);
hold on;
plot(X(ind_c2,3), X(ind_c2,4), 'xb','Linewidth',1);
plot(X_train1(:,3), X_train1(:,4), 'og','Linewidth',3);
plot(X_train2(:,3), X_train2(:,4), 'xc','Linewidth',3);
xLimits = get(gca,'XLim');  %# Get the range of the x axis
yLimits = get(gca,'YLim');
plot(X_utrain(:,3)', ( -w_t_2(1)*X_utrain(:,3)')./w_t_2(2), 'b', 'Linewidth',2 );
plot(X_utrain(:,3)', ( -w_s_2(1)*X_utrain(:,3)')./w_s_2(2), '--m', 'Linewidth',1.5 );

%plot(X_utrain(:,3)', ( -w(3)*X_utrain(:,3)')./w(4), 'g', 'Linewidth',2 );

%plot(X_utrain(:,3)', ( -w20(1)*X_utrain(:,3)')./w20(2), '-.m', 'Linewidth',1 );
grid on;
axis([xLimits,yLimits]);
hold off
legend('c1-unlabeled','c2-unlabeled','c1-labeled','c2-labeled','multi-view', 'single-view')
title('view 2')
pause(1)


