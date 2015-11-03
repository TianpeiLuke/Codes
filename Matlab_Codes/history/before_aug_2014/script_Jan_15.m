%%find a random split with equal performance

clear all
close all
clc
addpath('../../../../../../MATLAB/cvx/');
addpath('../libsvm-3.16/matlab/')
% fid = fopen('./mushrooms','r');
% 
% 
% fclose(fid)


[label_vector, instance_matrix] = libsvmread('mushrooms');


run = 1;
error_c = zeros(2,run);
error_d = zeros(2,run);

Ntr    = 100;
Nutr   = 2000;
Ntest  = 4000;

load('fea_Jan_15.mat')
%ind_test = randsample(size(instance_matrix,1), Ntest);
%ind_tr   = randsample(setdiff([1:size(instance_matrix,1)], ind_test), Ntr);
%ind_utr   = randsample(setdiff([1:size(instance_matrix,1)], union(ind_test, ind_tr)), Nutr);

r = 1;
T = 3;%40;

history_Kll = cell(2,T);
history_Klu = cell(2,T);
history_w = cell(2,T);
history_q = cell(1,T+1);
history_KL = zeros(2,T);
history_hdist = zeros(1,T);
history_phi = cell(1,T+1);
history_error = zeros(2,T);
history_pred_dif = cell(1,T);

options.Kernel = 'rbf';
options.KernelParam = sqrt(1/(2*1));



flag = 0;
epsilon = 0.3;
iter = 0;
while(flag == 0)
iter = iter + 1;
iter
%fea_v1 = randsample(size(instance_matrix,2), 50);
%fea_v2 = randsample(setdiff([1:size(instance_matrix,2)], fea_v1)', 50);

label_vector_1 = (label_vector - 1.5)*2;





X_test_v1 =  [instance_matrix(ind_test, fea_v1)];% ones(Ntest,1)];
X_test_v2 =  [instance_matrix(ind_test, fea_v2)];% ones(Ntest,1)];
y_test    =  label_vector_1(ind_test);

X_train_v1 = [instance_matrix(ind_tr, fea_v1)];% ones(Ntr,1)];
X_train_v2 = [instance_matrix(ind_tr, fea_v2)];% ones(Ntr,1)];
y_train    =  label_vector_1(ind_tr);

X_utrain_v1 = [instance_matrix(ind_utr, fea_v1)];% ones(Nutr,1)];
X_utrain_v2 = [instance_matrix(ind_utr, fea_v2)];% ones(Nutr,1)];
y_utrain   =  label_vector_1(ind_utr);

display('Compute the kernel matrices...')
K_ll_1 = (y_train*y_train').*calckernel(options,X_train_v1);
K_ll_2 = (y_train*y_train').*calckernel(options,X_train_v2);


history_Kll{1,1} =  K_ll_1;
history_Kll{2,1} =  K_ll_2;

%K_lu_1 = calckernel(options,X_utrain_v1,X_train_v1);
%K_lu_2 = calckernel(options,X_utrain_v2,X_train_v2);

K_tl_1 = calckernel(options,X_test_v1,X_train_v1);
K_tl_2 = calckernel(options,X_test_v2,X_train_v2);

lin_vec_v1 =  - ones(Ntr, 1);
lin_vec_v2 =  - ones(Ntr, 1);
q = 0.5*ones(Nutr, 1);

%% compare with single view 
cvx_begin sdp 
   variable alpha_s1(Ntr,1) nonnegative
   maximize (-0.5*alpha_s1'*(K_ll_1)*(alpha_s1) + ones(Ntr,1)'*alpha_s1) 
   subject to
     %alpha_s1'*y_train == 0 ;
     alpha_s1 <= ones(Ntr,1);
cvx_end

cvx_begin sdp 
   variable alpha_s2(Ntr,1) nonnegative
   maximize (-0.5*alpha_s2'*(K_ll_2)*(alpha_s2) + ones(Ntr,1)'*alpha_s2) 
   subject to
     %alpha_s2'*y_train == 0 ;
     alpha_s2 <= ones(Ntr,1);
cvx_end

%f1 = sum(diag(y_train.*alpha_s1)*K_lu_1,1)';
%f2 = sum(diag(y_train.*alpha_s2)*K_lu_2,1)'; 

y_pred_v1 = sign(sum(diag(y_train.*alpha_s1)*K_tl_1,1))';
y_pred_v2 = sign(sum(diag(y_train.*alpha_s2)*K_tl_2,1))';

error_v1 = length(find(y_test~=y_pred_v1))/length(y_test)*100;
error_v2 = length(find(y_test~=y_pred_v2))/length(y_test)*100;
error_c(1,r) = error_v1;
error_c(2,r) = error_v2;

% model1 = svmtrain(y_train, X_train_v1, '-c 1 -g 0.07 ');
% model2 = svmtrain(y_train, X_train_v2, '-c 1 -g 0.07 ');
% 
% [predict_label1, accuracy1,~] = svmpredict(y_test, X_test_v1, model1);
% [predict_label2, accuracy2,~] = svmpredict(y_test, X_test_v2, model2);

 if abs(error_v1 - error_v2) < epsilon
    flag = 1;
 end
end


isrun =1;


if isrun ==1
    %%



%history_hdist = cell(1,T);

% compute the dual variable




%w_s_1 = (sum(diag(y_train.*alpha_s1)*X_train(:,[1:2]), 1))' ;
%w_s_2 = (sum(diag(y_train.*alpha_s2)*X_train(:,[3:4]), 1))' ;



%model1 = svmtrain(y_train, X_train_v1, '-c 1 -g 0.07 ');
%model2 = svmtrain(y_train, X_train_v2, '-c 1 -g 0.07 ');

%[predict_label1, accuracy1,~] = svmpredict(y_test, X_test_v1, model1);
%[predict_label2, accuracy2,~] = svmpredict(y_test, X_test_v2, model2);


%f1 = sum(diag(y_train.*alpha_s1)*K_lu_1,1)';
%f2 = sum(diag(y_train.*alpha_s2)*K_lu_2,1)'; 



%q = 1./(1 + exp(-0.5*f1/norm(f1,inf)- 0.5*f2/norm(f2,inf)));
history_q{1} = q;

options1.Kernel = 'rbf';
options1.KernelParam = sqrt(1/(2*1));

options2.Kernel = 'rbf';
options2.KernelParam = sqrt(1/(2*1));

K_ll_1 = (y_train*y_train').*calckernel(options1,X_train_v1);
K_ll_2 = (y_train*y_train').*calckernel(options2,X_train_v2);

K_lu_1 = calckernel(options1,X_utrain_v1,X_train_v1);
K_lu_2 = calckernel(options2,X_utrain_v2,X_train_v2);

K_uu_1 = calckernel(options1,X_utrain_v1);
K_uu_2 = calckernel(options2,X_utrain_v2);

K_tl_1 = calckernel(options1,X_test_v1,X_train_v1);
K_tl_2 = calckernel(options2,X_test_v2,X_train_v2);

K_tu_1 = calckernel(options1,X_test_v1,X_utrain_v1);
K_tu_2 = calckernel(options2,X_test_v2,X_utrain_v2);

%% multiview with pseudo-label 

for t=1:T
%% construct the MED classifier
display(['========================================================'])
display(sprintf('Iter %d', t))
display('Construct semi-supervised MED learner')
y_utrain_psl = (q - 0.5*ones(Nutr, 1));
lin_vec_v1 =  -(y_train*y_utrain_psl').*K_lu_1*ones(Nutr,1) + ones(Ntr, 1);   
lin_vec_v2 =  -(y_train*y_utrain_psl').*K_lu_2*ones(Nutr,1) + ones(Ntr, 1);   
    
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

%w_t_1 = (sum(diag(y_train.*alpha_v1)*X_train(:,[1:2]), 1))' + (sum(diag(q - 0.5*ones(Nutr, 1))*X_utrain(:,[1:2]),1))';
%w_t_2 = (sum(diag(y_train.*alpha_v2)*X_train(:,[3:4]), 1))' + (sum(diag(q - 0.5*ones(Nutr, 1))*X_utrain(:,[3:4]),1))';


display('Decision making...')
f1 = sum(diag(y_train.*alpha_v1)*K_lu_1,1)' +...
    sum(diag(y_utrain_psl)*K_uu_1,1)';
f2 = sum(diag(y_train.*alpha_v2)*K_lu_2,1)' +...
    sum(diag(y_utrain_psl)*K_uu_2,1)';

history_w{1,t} = sign(f1);
history_w{2,t} = sign(f2);
history_pred_dif{1,t} = abs(sign(f1) - sign(f2))/2;
%% step 2: allocate the pseudo-label
% display(sprintf('Iter %d Compute psudo-label', t));
% q = 1./(1 + exp(-0.5*1.5*(f1/norm(f1,inf)+f2/norm(f2,inf))));
% history_q{t+1} = q; 
% 
% if max(abs(q- history_q{t})) < 1e-3 && t>2
%     break;
% end


%
%% step 2: allocate the pseudo-label
display(sprintf('Iter %d Compute psudo-label', t));
sigma = 0.5;
% q = 1./(1 + exp(-2*(sigma*f1/norm(f1,inf)+(1-sigma)*f2/norm(f2,inf))));
 q = (y_utrain+1)/2;
history_q{t+1} = q; 



% if max(abs(q- history_q{t})) < 1e-3 && t>2
%     break;
% end

%% 
% p1 = 1./(1 + exp(-f1));
% p2 = 1./(1 + exp(-f2));
% p  = 0.5*(y_utrain+ ones(Nutr,1));
% 
% KL_1 = sum(filt(q.*log(q./p1)) + filt((ones(Nutr,1)-q).*log((ones(Nutr,1)-q)./(ones(Nutr,1)-p1))));
% KL_2 = sum(filt(q.*log(q./p2)) + filt((ones(Nutr,1)-q).*log((ones(Nutr,1)-q)./(ones(Nutr,1)-p2))));
% %h_dist = 1/sqrt(2)*sqrt((sqrt(p1) - sqrt(p2)).^2 + (sqrt(ones(Nutr,1)-p1) - sqrt(ones(Nutr,1)-p2)).^2);
% %b_dist = -log(sqrt(p1.*p2) + sqrt((ones(Nutr,1)-p1).*(ones(Nutr,1)-p2)));
% h_dist = 1/sqrt(2)*sum(sqrt((sqrt(q) - sqrt(p)).^2 + (sqrt(ones(Nutr,1)-q) - sqrt(ones(Nutr,1)-p)).^2))/Nutr;
% 
% 
% history_hdist(t) = h_dist;
% history_KL(1,t)=KL_1; 
% history_KL(2,t)=KL_2; 
%%
display(sprintf('Iter %d Compute phi', t));
phi_v1 =  0.5*sign(q - ones(Nutr, 1)).*...
        (-f1 + f2)...
        - log(1 + exp(-0.5*f1/norm(f1,inf)- 0.5*f2/norm(f2,inf)))...
       +  log(1 + exp(- f1/norm(f1,inf) )) ;

phi_v2 =  0.5*sign(q - ones(Nutr, 1)).*...
        (f1 - f2)...
        - log(1 + exp(-0.5*f1/norm(f1,inf)- 0.5*f2/norm(f2,inf)))...
       +  log(1 + exp(- f1/norm(f1,inf) )) ;   
   
   
history_phi{t+1} = min([phi_v1,phi_v2],[],2);

%%
display(sprintf('Iter %d Prediction', t));
f1_tst = sum(diag(y_train.*alpha_v1)*K_tl_1,1)' +...
    sum(diag(y_utrain_psl)*K_tu_1,1)';
f2_tst = sum(diag(y_train.*alpha_v2)*K_tl_2,1)' +...
    sum(diag(y_utrain_psl)*K_tu_2,1)';

y_pred_c1 = sign(f1_tst);
y_pred_c2 = sign(f2_tst);

history_error(1,t)= length(find(y_test~= y_pred_c1))/length(y_test)*100;
history_error(2,t)= length(find(y_test~= y_pred_c2))/length(y_test)*100;



end

display('Prediction ...')
f1_tst = sum(diag(y_train.*alpha_v1)*K_tl_1,1)' +...
    sum(diag(y_utrain_psl)*K_tu_1,1)';
f2_tst = sum(diag(y_train.*alpha_v2)*K_tl_2,1)' +...
    sum(diag(y_utrain_psl)*K_tu_2,1)';

y_pred_c1 = sign(f1_tst);
y_pred_c2 = sign(f2_tst);

error_vc1 = length(find(y_test~= y_pred_c1))/length(y_test)*100;
error_vc2 = length(find(y_test~= y_pred_c2))/length(y_test)*100;

%error_v1
%error_v2

error_d(1,r) = error_vc1;
error_d(2,r) = error_vc2;

%%  two single view with pseudo-label
display('two single view on all training data')
y_aug_train = [y_train; sign(q - 0.5)]; %[y_train; y_utrain]; 
X_aug_train_v1 = [X_train_v1; X_utrain_v1]; 
X_aug_train_v2 = [X_train_v2; X_utrain_v2]; 

K_aug_1 = (y_aug_train*y_aug_train').*calckernel(options,X_aug_train_v1);
K_aug_2 = (y_aug_train*y_aug_train').*calckernel(options,X_aug_train_v2);
K_al_1 = calckernel(options1,X_test_v1,X_aug_train_v1);
K_al_2 = calckernel(options2,X_test_v2,X_aug_train_v2);


cvx_begin sdp 
   variable alpha_a1(Ntr+Nutr,1) nonnegative
   maximize (-0.5*alpha_a1'*(K_aug_1)*(alpha_a1) + ones(Nutr+Ntr,1)'*alpha_a1) 
   subject to
     %alpha_s1'*y_train == 0 ;
     alpha_a1 <= ones(Nutr+Ntr,1);
cvx_end

cvx_begin sdp 
   variable alpha_a2(Ntr+Nutr,1) nonnegative
   maximize (-0.5*alpha_a2'*(K_aug_2)*(alpha_a2) + ones(Nutr+Ntr,1)'*alpha_a2) 
   subject to
     %alpha_s2'*y_train == 0 ;
     alpha_a2 <= ones(Nutr+Ntr,1);
cvx_end

%f1 = sum(diag(y_train.*alpha_s1)*K_lu_1,1)';
%f2 = sum(diag(y_train.*alpha_s2)*K_lu_2,1)'; 

y_pred_a1 = sign(sum(diag(y_aug_train.*alpha_a1)*K_al_1,1))';
y_pred_a2 = sign(sum(diag(y_aug_train.*alpha_a2)*K_al_2,1))';

error_a1 = length(find(y_test~=y_pred_a1))/length(y_test)*100;
error_a2 = length(find(y_test~=y_pred_a2))/length(y_test)*100;
error_au(1,r) = error_a1;
error_au(2,r) = error_a2;

%%
X_aug_train = [X_train_v1, X_train_v2; X_utrain_v1, X_utrain_v2]; 

K_aug = (y_aug_train*y_aug_train').*calckernel(options,X_aug_train);

K_al = calckernel(options1,[X_test_v1, X_test_v2],X_aug_train);



cvx_begin sdp 
   variable alpha_a(Ntr+Nutr,1) nonnegative
   maximize (-0.5*alpha_a'*(K_aug)*(alpha_a) + ones(Nutr+Ntr,1)'*alpha_a) 
   subject to
     %alpha_s1'*y_train == 0 ;
     alpha_a <= ones(Nutr+Ntr,1);
cvx_end



%f1 = sum(diag(y_train.*alpha_s1)*K_lu_1,1)';
%f2 = sum(diag(y_train.*alpha_s2)*K_lu_2,1)'; 

y_pred_a = sign(sum(diag(y_aug_train.*alpha_a)*K_al,1))';


error_a = length(find(y_test~=y_pred_a))/length(y_test)*100;

error_au_c(1,r) = error_a;






%%
q_hs = cell2mat(history_q);

figure(1)
plot(1:t-1, history_error(1,1:(t-1)),'b');
hold on
plot(1:t-1, history_error(2,1:(t-1)),'r');
%plot(1:t-1, error_v1*ones(1,t-1),'-*c');
%plot(1:t-1, error_v2*ones(1,t-1),'-*m');
plot(1:t-1, error_a1*ones(1,t-1),'--b');
plot(1:t-1, error_a2*ones(1,t-1),'--r');
plot(1:t-1, error_a*ones(1,t-1),'-*m');
xlabel 'iteration'
ylabel 'test error (\%)'
legend('view 1', 'view 2', 'single view 1', 'single view 2', 'single view concatenated')
%legend('initial view 1', 'initial view 2','single view 1', 'single view 2')%, 'single view concatenated')

disagr = zeros(1,t-1);
for s=1:t-1
   disagr(s) =  length(find(history_pred_dif{1,s}))/length(history_pred_dif{1,s});
end

figure(2)
plot(1:t-1, disagr*100, 'r');
xlabel 'iter'
ylabel 'disagreement rate (\%)'
end



