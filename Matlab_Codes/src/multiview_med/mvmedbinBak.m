function [accuracy, errorlist, dev_tst, prob_tst, dev_trn, prob_trn, ...
           history, history_tst,programflag] = mvmedbinBak(Traindata, Testdata, param, iniparam) 
%% function that learns the multiview binary MED classifier using MED and Bhattcharrya distance
% Input: 
%   Traindata:  a struct for training set
%            .nV:  no. of views
%            .nU:  no. of unlabeled samples
%            .nL:  no. of labeleed samples
%            .d:   dimension of features
%            .X_U: nU x d x nV data set for unlabeled data
%            .X_L: nL x d x nV data set for labeled data
%            .y_L: nL x 1 labels 
%            .y_U: nU x 1 the ground truth for unlabeled data (not used in learning)
%   Testdata:  a struct for test set
%            .nTst:  no. of unlabeled samples
%            .d:   dimension of features
%            .X_Tst: nTst x d x nV data set for test data
%            .y_Tst: nTst x 1 the ground truth for test data (for error estimate)
%     param:  a struct for parameters
%            .kernelMethod:  
%                     'linear' for linear kernel
%                     'rbf' for Gaussian RBF kernel
%                     'poly' for polynominial kernel
%            .kernelParm:
%                       if 'linear', no need
%                   elseif 'rbf', 
%                       for variance sigma_k
%                   elseif 'poly' 
%                       for degree of polynominal k and bias term b
%            .regParam:     parameter for B-distance regularization
%            .maxIterOut:  maximum iteration for the outer loop
%            .threOut   :  stopping threshold for the outer loop
%            .maxIterMAP:  maximum iteration for MAP computing
%            .threMAP:     stopping threshold for MAP computing
%            .sigmaPri:    sigma for prior of weights
%            .mode:    =1; for normal mode
%                      =0; no accuracy computed
%   iniparam:    a struct for initial parameters
%            .inimodel: 1 x nV cell for initial model computed 
%
%
% Output:
%   accuracy: expected accuracy under consensus q(y|x1,x2)
%   dev_tst:  nTst x nV+1 decision value for test samples; the last column
%             is from consensus view
%
%  prob_tst:  nTst x nV+1, probability on test samples; the last column is 
%              consensus probability on test samples
% 
%
%   dev_trn:  a struct for decision values
%          .f_trn_u:  nU x nV decision value for unlabeled samples
%          .f_trn_l:  nL x nV decision value for labeled samples
%  prob_trn:  nU x 1, (consensus) probability values on unlabeled
%              samples
%          
%  history:  a struct for tracking the training procedure
%  history_tst: a struct for tracking the testing procedure
%
% Written by Tianpei Xie, Sep 10. 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../../../../../../MATLAB/cvx/');

nU = Traindata.nU;
nL = Traindata.nL;
nV = Traindata.nV;
nTst = Testdata.nTst;

maxIter = param.maxIterOut;
sigma2 = (param.sigmaPri)^2;
epsilon = param.threMAP;

model = iniparam.inimodel;

regParam = param.regParam;
options.Kernel = param.kernelMethod;
options.KernelParam =param.kernelParm;

X_U = Traindata.X_U;
y_U = Traindata.y_U;
X_L = Traindata.X_L;
y_L = Traindata.y_L;

X_Tst = Testdata.X_Tst;
y_Tst = Testdata.y_Tst;

programflag = 1;

% compute the kernel matrix
K_u = cell(1,nV);
K_l = cell(1,nV);
K_ul = cell(1,nV);
K_utst = cell(1,nV);  % kernel for test samples
K_ltst = cell(1,nV);  
for i=1:nV
K_u{i} = calckernel(options,X_U(:,:,i));
K_l{i}=  calckernel(options,X_L(:,:,i));
K_ul{i} = calckernel(options,X_L(:,:,i),X_U(:,:,i)); %U*L'
K_utst{i} = calckernel(options,X_Tst(:,:,i),X_U(:,:,i)); %U*Tst' kernel for test samples
K_ltst{i} = calckernel(options,X_Tst(:,:,i),X_L(:,:,i)); %L*Tst'
end
sigmoid = @(x)(1./(1+ exp(-x))); % classifier for binary set

q = 0.5*ones(nU,1);  % pseudo-label probablity on unlabeled set
q_tst = 0.5*ones(nTst, 1); 
errorlist = zeros(nTst, 1);
accuracy = 0;

fl0 = zeros(nL,nV);  % initial decision value on labeled set
fu0 = zeros(nU,nV);  %                   ...  on unlabeled set
ftst0 = zeros(nTst,nV); % initial decision value on test set

dev_tst = zeros(nTst,nV+1);
prob_tst = zeros(nTst,nV+1);
%%----------------------- history tracking---------------------
v_history = zeros(nU,nV,maxIter+1);
q_history = zeros(nU, maxIter+1);
dev_history = zeros(nU,nV, maxIter+1);
dual_history = zeros(nL,nV, maxIter+1);
p_history = zeros(nU,nV, maxIter+1);
fmap_history = zeros(nU, nV, maxIter+1);
fpred_history = zeros(nL, nV, maxIter+1);
fjointu_history = zeros(nU, nV, maxIter+1);
fjointl_history = zeros(nL, nV, maxIter+1);

fmap_tst_history= zeros(nTst, nV, maxIter+1);
fjoint_tst_history= zeros(nTst, nV, maxIter+1);
q_tst_history = zeros(nTst, maxIter+1);
%% -------------------- Initialization ----------------------
 % compute the initial value of SVM decision
display(sprintf('\n============================================'));
display(sprintf('Initializing...'));
for i=1:nV
  Kl_temp = calckernel(options,full(model{i}.SVs),X_L(:,:,i));
  fl0(:,i) = Kl_temp*model{i}.sv_coef;
  Ku_temp = calckernel(options,full(model{i}.SVs),X_U(:,:,i));
  fu0(:,i) = Ku_temp*model{i}.sv_coef;
  % SVM Prediction
  Ktst_temp = calckernel(options,full(model{i}.SVs),X_Tst(:,:,i));
  ftst0(:,i) = Ktst_temp*model{i}.sv_coef;
end

view_weight = 0.5*ones(nV,1);
q= sigmoid(fu0*view_weight); %compute the averge prediction 
q_tst= sigmoid(ftst0*view_weight); %compute the averge prediction 

q_history(:,1) = q;
dev_history(:,:,1) = fu0;
q_tst_history(:,1) = q_tst;

 % find the MAP estimate for unlabeled part of decision
fmap = fu0;  %map estimate on unlabeled data
fpred = fl0; %map estimate on labeled data
ftst_map = ftst0;   %map estimate on test data

fjointu = fmap;
fjointl = fpred;
fjoint_tst = ftst_map;
% update M for Hessian matrix 
v = zeros(nU, nV);
M = zeros(nU, nU, nV);
for i=1:nV
 v(:,i) = 1/2*1./(1+cosh(fmap(:,i)));    
 %v1 = 1/2*1./(1+cosh(w_int1'*[X_U(1:2,:);ones(1,size(X_U(1:2,:),2))]));
 M(:,:,i) = diag(v(:,i));
end
v_history(:,:,1) = v;
p =  0.5*ones(nU,nV);
p_history(:,:,1) = p;
fmap_history(:,:,1) = fmap;
fpred_history(:,:,1) = fpred;
fjointu_history(:,:,1) = fjointu;
fjointl_history(:,:,1) = fjointl;

fmap_tst_history(:,:,1)= ftst0;
fjoint_tst_history(:,:,1) = fjoint_tst;

%track the probablity 
track_diff = zeros(nV,param.maxIterMAP);

%% ------------ outer loop for EM-style learning ---------------
iout = 0;
outflag = 1;
while( outflag && iout< maxIter)
 iout = iout+1;
 display(sprintf('outer loop: i=%d \n =============================================',iout));
 q_pre = q;
 
 regParam = 0.5*(1- 1/sqrt(iout+1)); %regParam -> 1
 
 t= 0;
 flag_w1 = 1;
 dual_alpha = zeros(nL, nV);
 P = zeros(nL, nL,  nV);
 U = zeros(nL, nL,  nV);
 S = zeros(nL, nL,  nV);
 %initalization of fmap
 fmap = fu0;  %map estimate on unlabeled data
 fpred = fl0; %map estimate on labeled data
 ftst_map = ftst0;
 epsilon_t =  5e-4;
 %% --------- inner loop for MAP update using Newton's method --------------
  while(flag_w1 && t<param.maxIterMAP)
  t = t+1; 
  display(sprintf('-- inner loop: t=%d',t));

    if sum(sum(isnan(fmap)))>0
           dvec = datevec(now);
          save(sprintf('error_%d%d%d%d%d.mat',dvec(1),dvec(2),dvec(3),dvec(4),dvec(5)))
          error(sprintf('error at %s', datestr(now)))
    end
  
  
   % update p
    for i=1:nV
        p(:,i) =  sigmoid(fmap(:,i));  
    end
    
    mu = ones(1,nV); % dumping factor
    fmap_temp = zeros(nU, nV);
    v_temp = v;
    M_temp = M;
    p_temp = p;
    
%%   % update f_map1, f_map2
    f_pre = fmap;
    %w_pre1 = w_map1;
    % historyw1(:,t)= w_map1;
    % historydiff(t)= sum(abs(q - p1));
    % historydiff2(t)= sum(abs(q - p2));
    for i=1:nV
      if sum(isnan(diag(M(:,:,i))))>0 || sum(isinf(diag(M(:,:,i))))>0
        track_diff(:,t+1:end) = [];
        dvec = datevec(now);
        save(sprintf('error_%d%d%d%d%d.mat',dvec(1),dvec(2),dvec(3),dvec(4),dvec(5)))
        error(sprintf('error at %s', datestr(now)))
      end
      % --- tune the dumping factor ---
      idump = 0;
      %display('tune dumping factor')
      ncount = 0;
      while(~idump)
       fmap_temp(:,i) = (1-mu(i))*fmap(:,i)+ ...
          mu(i)*regParam*K_u{i}*((1/sigma2*pinv(M(:,:,i)) + regParam*K_u{i})\fmap(:,i))...
          + mu(i)*regParam*sigma2*K_u{i}*(q-p(:,i)) ...
         - mu(i)*regParam^2*sigma2*K_u{i}*((1/sigma2*pinv(M(:,:,i)) +...
                                     regParam*K_u{i})\K_u{i}*(q-p(:,i)) );
     
       v_temp(:,i) = 1/2*1./(1+cosh(fmap_temp(:,i)));    
       M_temp(:,:,i) = diag(v(:,i));
       p_temp(:,i) =  sigmoid(fmap_temp(:,i));  
       
       ratio = norm(fmap_temp(:,i) - sigma2*regParam*K_u{i}*(q-p_temp(:,i)))...
           /norm(fmap(:,i) - sigma2*regParam*K_u{i}*(q-p(:,i)));
       
       if 1- ratio > epsilon_t || ratio == 1
         idump = 1;  
       end
       ncount = ncount+1;
       
       mu(i) = mu(i)/2; %dumping factor decreases 
       if ncount>1
        display(sprintf('count: %d',ncount));
        display(sprintf('mu: %e\n',mu(i)));
       end
       if ncount > 200
           dvec = datevec(now);
          save(sprintf('error_%d%d%d%d%d.mat',dvec(1),dvec(2),dvec(3),dvec(4),dvec(5)))
          display(sprintf('error at %s', datestr(now)))
            programflag = 0;
         accuracy = 0; 
      errorlist = []; 
      dev_tst = []; 
      prob_tst = []; 
      dev_trn = []; 
      prob_trn = []; 
      history.v_history = v_history;
      history.q_history = q_history;
        history.dev_history = dev_history;
        history.dual_history = dual_history;
        history.p_history = p_history;
        history.fmap_history = fmap_history;
        history.fpred_history = fpred_history;
        history.fjointu_history = fjointu_history;
        history.fjointl_history = fjointl_history;
        history_tst = [];
        return;
       end
      end
      %display('finish tuning')
      % ---- update ---------------------
      fmap(:,i) = (1-mu(i))*fmap(:,i)+ ...
          mu(i)*regParam*K_u{i}*((1/sigma2*pinv(M(:,:,i)) + regParam*K_u{i})\fmap(:,i))...
          + mu(i)*regParam*sigma2*K_u{i}*(q-p(:,i)) ...
          - mu(i)*regParam^2*sigma2*K_u{i}*((1/sigma2*pinv(M(:,:,i)) +...
                                    regParam*K_u{i})\K_u{i}*(q-p(:,i)) );
      
      fpred(:,i) = (1-mu(i))*fpred(:,i)+ ...
          mu(i)*regParam*K_ul{i}'*((1/sigma2*pinv(M(:,:,i)) + regParam*K_u{i})\fmap(:,i))...
          + mu(i)*regParam*sigma2*K_ul{i}'*(q-p(:,i)) ...
          - mu(i)*regParam^2*sigma2*K_ul{i}'*((1/sigma2*pinv(M(:,:,i)) +...
                                     regParam*K_u{i})\K_u{i}*(q-p(:,i)) );
      
      % MAP Prediction
      ftst_map(:,i) = (1-mu(i))*ftst_map(:,i)+ ...
          mu(i)*regParam*K_utst{i}'*((1/sigma2*pinv(M(:,:,i)) + regParam*K_u{i})\fmap(:,i))...
          + mu(i)*regParam*sigma2*K_utst{i}'*(q-p(:,i)) ...
          - mu(i)*regParam^2*sigma2*K_utst{i}'*((1/sigma2*pinv(M(:,:,i)) +...
                                      regParam*K_u{i})\K_u{i}*(q-p(:,i)) );
    end

%%   % update M
   for i=1:nV
      v(:,i) = 1/2*1./(1+cosh(fmap(:,i)));    
      M(:,:,i) = diag(v(:,i));
   end
   
   crit_vec = zeros(1,nV);
   for i=1:nV
      crit_vec(i) = (norm(fmap(:,i)-f_pre(:,i))<epsilon);
      track_diff(i,t) = norm(fmap(:,i)-f_pre(:,i));
   end
   
   if sum(crit_vec) == nV
            flag_w1 = 0;
            track_diff(:,t+1:end) = [];
    %    history1(t+1:end) = [];
    %    historydiff(t+1:end) = [];
    %    historydiff2(t+1:end) = [];
    %    historyw1(:,t+1:end) = [];
    %    loss1(:,t+1:end) = [];
    %    loss2(:,t+1:end) = [];
   end
   if t>500
       track_diff(:,t+1:end) = [];
       dvec = datevec(now);
       save(sprintf('error_%d%d%d%d%d.mat',dvec(1),dvec(2),dvec(3),dvec(4),dvec(5)))
       display(sprintf('error at %s', datestr(now)))
            programflag = 0;
      accuracy = 0; 
      errorlist = []; 
      dev_tst = []; 
      prob_tst = []; 
      dev_trn = []; 
      prob_trn = []; 
      history.v_history = v_history;
      history.q_history = q_history;
        history.dev_history = dev_history;
        history.dual_history = dual_history;
        history.p_history = p_history;
        history.fmap_history = fmap_history;
        history.fpred_history = fpred_history;
        history.fjointu_history = fjointu_history;
        history.fjointl_history = fjointl_history;
        history_tst = [];
        return;
   end
    fmap_temp = [];
    v_temp = [];
    M_temp = [];
    p_temp = [];
  end
  fmap_history(:,:,iout+1) = fmap;
  fpred_history(:,:,iout+1) = fpred;
  fmap_tst_history(:,:,iout+1)= ftst_map;
  
  for i=1:nV
     p(:,i) =  sigmoid(fmap(:,i));  
  end
  p_history(:,:,iout+1) = p;
  
  for i=1:nV
    v(:,i) = 1/2*1./(1+cosh(fmap(:,i)));    
    M(:,:,i) = diag(v(:,i));
  end
  v_history(:,:,iout+1) = v;
  
  %% find sparse dual variables for labeled part
 len = size(X_L,1);
 b = zeros(len,nV);
 %eones = ones(len, 1);
 rho = 1.1+1.5*rand(1);
 param_qu.sigma2 = sigma2 ;
 for ii=1:nV
  b(:,ii) = fpred(:,ii).*y_L;
  P(:,:,ii) =  (rho*K_l{ii} - regParam*(K_ul{ii}'*((pinv(M(:,:,ii))/sigma2 +...
                                regParam*K_u{ii})\K_ul{ii}))).*(y_L*y_L');
  [U(:,:,ii), S(:,:,ii)] = eig(P(:,:,ii));
  % call qudratic-solver to solve it
  [alpha, status] = quadr_svm(0.5*(P(:,:,ii)+P(:,:,ii)'), b(:,ii),param_qu);
%   cvx_begin sdp
%    cvx_precision high
%    variable alpha(len,1) nonnegative;
%    %minimize(-(eones-b1)'*alpha1 + 0.5*alpha1'*P1*alpha1 )
%    minimize(-(eones- b(:,ii))'*alpha + 0.5*alpha'*(U(:,:,ii)*S(:,:,ii)*U(:,:,ii)')*alpha )
%      subject to
%        alpha <= 1;
%   cvx_end 
  if strcmp(status,'Solved') || strcmp(status,'Inaccurate/Solved')
  dual_alpha(:,ii) = alpha;
  else
      display('Error: no solution')
      programflag = 0;
      accuracy = 0; 
      errorlist = []; 
      dev_tst = []; 
      prob_tst = []; 
      dev_trn = []; 
      prob_trn = []; 
      history.v_history = v_history;
      history.q_history = q_history;
        history.dev_history = dev_history;
        history.dual_history = dual_history;
        history.p_history = p_history;
        history.fmap_history = fmap_history;
        history.fpred_history = fpred_history;
        history.fjointu_history = fjointu_history;
        history.fjointl_history = fjointl_history;
        history_tst = [];
      return;
  end
 end
 

%%
for i=1:nV
    %unlabeled part
    fjointu(:,i) =  fmap(:,i) + sigma2*K_ul{i}*(y_L.*dual_alpha(:,i)) ...
                             - sigma2*regParam*K_u{i}*...
                             ((pinv(M(:,:,i))/sigma2 + regParam*K_u{i})...
                             \K_ul{i})*(y_L.*dual_alpha(:,i));
                         
    %lableled part
    fjointl(:,i) =  fpred(:,i)+ sigma2*K_l{i}*(y_L.*dual_alpha(:,i)) ...
                             - sigma2*regParam*K_ul{i}'*...
                             ((pinv(M(:,:,i))/sigma2 + regParam*K_u{i})...
                             \K_ul{i})*(y_L.*dual_alpha(:,i));
    
    %Prediction  
    fjoint_tst(:,i) =  ftst_map(:,i)+ sigma2*K_ltst{i}'*(y_L.*dual_alpha(:,i)) ...
                             - sigma2*regParam*K_utst{i}'*((pinv(M(:,:,i))/sigma2 ...
                             + regParam*K_u{i})\K_ul{i})*(y_L.*dual_alpha(:,i));
                         
 %f1(union(ind_U1, ind_U2)) =  X_U(1:2,:)'*w_map1 + K_ul1*(y_L.*alpha1) ...
 %                            - K_u1*((pinv(M1) + K_u1)\K_ul1)*(y_L.*alpha1);
 %f2(union(ind_U1, ind_U2)) =  X_U(3:4,:)'*w_map2 +  K_ul2*(y_L.*alpha2) ...
 %                             - K_u2*((pinv(M2) + K_u2)\K_ul2)*(y_L.*alpha2);
 %  f1(union(ind_L1, ind_L2)) =  X_L(1:2,:)'*w_map1 + K_l1*(y_L.*alpha1) ...
 %                              - K_ul1'*((pinv(M1) + K_u1)\K_ul1)*(y_L.*alpha1);
 % 
 % f2(union(ind_L1, ind_L2)) =  X_L(3:4,:)'*w_map2 + K_l2*(y_L.*alpha2) ...
 %                              - K_ul2'*((pinv(M2) + K_u2)\K_ul2)*(y_L.*alpha2);
end

fjointu_history(:,:,iout+1) = fjointu;
fjointl_history(:,:,iout+1) = fjointl;
fjoint_tst_history(:,:,iout+1) = fjoint_tst;

%% new consensus view
view_weight = 1/nV*ones(nV,1);
q= sigmoid(fjointu*view_weight); %compute the averge prediction 
q_tst = sigmoid(fjoint_tst*view_weight); 


%  fmap = fu0;  %map estimate on unlabeled data
%  fpred = fl0; %map estimate on labeled data
%  ftst_map = ftst0;

q_history(:,iout+1) = q; 
q_tst_history(:,iout+1) = q_tst;

dual_history(:,:,iout) =  dual_alpha;

if norm(q - q_pre)/nU < param.threOut
    outflag = 0;
    q_tst_history(:,iout+2:end) = [];
    v_history(:,:,iout+2:end) = [];
    dev_history(:,:,iout+2:end)= [];
    dual_history(:,:,iout+1:end) = [];
    p_history(:,:,iout+2:end)= [];
    fmap_history(:,:,iout+2:end)= [];
    fpred_history(:,:,iout+2:end)= [];
    fjointu_history(:,:,iout+2:end)= [];
    fjointl_history(:,:,iout+2:end)= [];


    q_tst_history(:,iout+2:end)= [];
    fmap_tst_history(:,:,iout+2:end) = [];
    fjoint_tst_history(:,:,iout+2:end) = [];
end
    


end
display('End of iteration');
%% Prediction
display('Prediction ...')
dev_tst(:,1:nV) = fjoint_tst;
dev_tst(:,nV+1) = fjoint_tst*view_weight;

if param.mode == 1
  errorlist = (sign(q_tst - 0.5*ones(nTst,1))~= y_Tst) ; 
  accuracy = 1 - sum(errorlist)/nTst;
  display(sprintf('Accuracy: %.2f %%', accuracy*100));
elseif param.mode == 0
    errorlist = [];
    accuracy  = -1;
    display('Prediction Ends')
end

%%
dev_trn.f_trn_u = fjointu;
dev_trn.f_trn_l = fjointl;
       prob_trn = q;
       
for ivv = 1:nV
   prob_tst(:,ivv) = sigmoid(fjoint_tst(:,ivv)); 
end
prob_tst(:,nV+1) = q_tst;
       
history.v_history = v_history;
history.q_history = q_history;
history.dev_history = dev_history;
history.dual_history = dual_history;
history.p_history = p_history;
history.fmap_history = fmap_history;
history.fpred_history = fpred_history;
history.fjointu_history = fjointu_history;
history.fjointl_history = fjointl_history;


history_tst.q_tst_history = q_tst_history;
history_tst.fmap_tst_history = fmap_tst_history;
history_tst.fjoint_tst_history = fjoint_tst_history;
display(sprintf('============================================\n'))
end



function K = calckernel(options,X1,X2)
% {calckernel} computes the Gram matrix of a specified kernel function.
% 
%      K = calckernel(options,X1)
%      K = calckernel(options,X1,X2)
%
%      options: a structure with the following fields
%               options.Kernel: 'linear' | 'poly' | 'rbf' 
%               options.KernelParam: specifies parameters for the kernel 
%                                    functions, i.e. degree for 'poly'; 
%                                    sigma for 'rbf'; can be ignored for 
%                                    linear kernel 
%      X1: N-by-D data matrix of N D-dimensional examples
%      X2: (it is optional) M-by-D data matrix of M D-dimensional examples
% 
%      K: N-by-N (if X2 is not specified) or M-by-N (if X2 is specified)
%         Gram matrix
%
% Author: Stefano Melacci (2009)
%         mela@dii.unisi.it
%         * based on the code of Vikas Sindhwani, vikas.sindhwani@gmail.com

kernel_type=options.Kernel;
kernel_param=options.KernelParam;

n1=size(X1,1);
if nargin>2
    n2=size(X2,1);
end

 switch kernel_type

    case 'linear'
        if nargin>2
            K=X2*X1';
        else
            K=X1*X1';
        end

    case 'poly'
        if nargin>2
            K=(X2*X1').^kernel_param;
        else
            K=(X1*X1').^kernel_param;
        end

    case 'rbf'
        if nargin>2
            K = exp(-(repmat(sum(X1.*X1,2)',n2,1) + ...
                repmat(sum(X2.*X2,2),1,n1) - 2*X2*X1') ...
                /(2*kernel_param^2));
        else
            P=sum(X1.*X1,2);
            K = exp(-(repmat(P',n1,1) + repmat(P,1,n1) ...
                - 2*X1*X1')/(2*kernel_param^2));
        end

    otherwise
        
       error('Unknown kernel function.');
 end
end



