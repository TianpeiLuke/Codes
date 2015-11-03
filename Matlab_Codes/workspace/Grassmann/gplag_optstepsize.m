function [X, history, epsilon] = gplag_optstepsize(X0, Y, Ky, param, option, sel)
%% conjugate gradient on Grassmann manifold
% using the optimal stepsize by Dejiao
% Input: 
%   X0: N x k,  initial orthogonal matrix
%   Ky: N x N,  the observed Gram matrix Y*Y'
% param
%    .N: # of samples
%    .s: amient dimension of Y
%    .k: intrinsic dimension of X
%    .T: resolution for optimal step size search
%    .beta: 1/beta is the noise variance; change the condition number of Kx
%    .maxIter: maximum iteration of conjugate gradient step;
%
% option
%      .kernelMethod:  
%          'linear' for linear kernel
%          'rbf' for Gaussian RBF kernel
%       .kernelParm:
%           if 'linear', no need
%           elseif 'rbf', 
%               for variance sigma_k
%           
% Output:
%    X: N x k, optimal subbasis for latent subspace
% history
%    history_X: cell array for intermediate X
%    history_error
%  
% 
% written by Tianpei Xie, May 11 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N= param.N;
s= param.s;
k= param.k;
T= param.T;
beta = param.beta;
maxIter = param.maxIter;
param.pdt = 0;
alphaT = param.alphaT;
history.X = cell(1,maxIter+1);

history.objval = zeros(1,maxIter+1);
history.Gnorm = zeros(1,maxIter+1);
history.GSigma = cell(1,maxIter+1);
history.fl = cell(1,maxIter);
history.tl = cell(1,maxIter);
history.sv = zeros(k,maxIter+1);


if strcmp(option.kernelMethod, 'linear')
 history.epsilon = zeros(1,maxIter+1);   
end

[Us, Ss] = eig(Ky);
[~,Idx] = sort(diag(Ss),'descend');
Us = Us(:,Idx);

%% Initialization
Gm0 = grad_man_GPLM(Ky, X0, param, option);
H = -Gm0;
G=  Gm0;
G_pre = Gm0;
[U,S,V] = svd(H,0);
history.sv(:,1) = diag(S);
X = X0;
gamma = ones(k,1);

matparm.U{1} = U;
matparm.V{1} = V;
matparm.S{1} = S;
history.GSigma{1} = S;

history.X{1} = X;

if strcmp(option.kernelMethod, 'linear')
%% compute the principal surface
R = X'*Y;
[UR,SR, VR] = svd(R);
Xtst = X*UR;
epsilon = k-norm(Xtst'*Us(:,1:k),'fro'); %measure the distance btw subspaces
history.epsilon(1) = epsilon;
end
objval = logP(Ky, X, param, option); 
history.objval(1) = objval;
history.Gnorm(1) = norm(H,'fro')/sqrt(N*k);
for t=1:maxIter
   history.iter = t+1; 
   display(sprintf('iter = %d; neg log-likelihood: %.5f; subspace-dist: %.2f',t,objval,epsilon)) 
   %% move on geodesic of Grassmannian
   display('compute step-size..')
   param.time = t;
   %[tmin] = step_size(X, Ky, matparm, param, option, history);
   Eta= opt_size(Ky, X, Y,U, param);
   Xnext=  X*V*diag(cos(diag(Eta)))*V' + U*diag(sin(diag(Eta)))*V'; 
   %% Find gamma
   
   
   %a = alphaT; 
   %rr = 1/4; %2/3;
   %tmin = a/(t+1).^rr;
   %Xnext=  X*V*diag(cos(tmin*diag(S)))*V' + U*diag(sin(tmin*diag(S)))*V'; 
   
   %% compute natural gradient on geodesic at Xt+1
   
   Gm = grad_man_GPLM(Ky, Xnext, param, option);
   history.Gnorm(t+1) = norm(H,'fro')/sqrt(N*k);
   display(sprintf('the norm of gradient: %.2f',norm(Gm,'fro')/sqrt(N*k)))
   if((mean(history.Gnorm([max([t-5,1]):t+1])) <param.thres_norm ) && param.dostop)
       return
   end
   
   %% gradient adjustment via parallel translation from Xt to Xt+1
   if sel == 2
   tH = -X*V*diag(sin(tmin*diag(S)))*S*V' + U*diag(cos(tmin*diag(S)))*S*V';
   tG = G - (X*V*diag(sin(tmin*diag(S)))+ U*(eye(k) - diag(cos(tmin*diag(S)))))*U'*G; 
  
   %% update conjugate gradient 
    rho = trace((Gm- tG)'*Gm)/trace(G_pre'*G_pre); 
    H = -Gm+ rho*tH;
    if (mod((k+1),k*(N-k)) == 0)
     H = -Gm;
    end
   else
        H = -Gm;
   end
   

    
   %% SVD of conjugate gradient  
   [U,S,V] = svd(H,0); 
   history.GSigma{t+1} = S;
   history.sv(:,t+1) = diag(S);
   %% update Xt to Xt+1
   X = Xnext;
   history.X{t+1} = X;
   if strcmp(option.kernelMethod, 'linear')
   %% compute the principal surface
   R = X'*Y;
   [UR,SR, VR] = svd(R);
   Xtst = X*UR;
   epsilon = k-norm(Xtst'*Us(:,1:k),'fro');
   history.epsilon(t+1) = epsilon;
   end
   
   objval = logP(Ky, X, param, option); 
   history.objval(t+1) = objval;
   matparm.U{1} = U;
   matparm.V{1} = V;
   matparm.S{1} = S;
   if ((epsilon <= param.thresh)&& param.dostop)
       return;
   end
end







