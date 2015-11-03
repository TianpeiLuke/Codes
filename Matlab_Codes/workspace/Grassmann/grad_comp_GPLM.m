function [Ge] = grad_comp_GPLM(Ky, X, param, option)
%% compute the Euclidean gradient of log p(Y|X) at X
% log p(Y|X) \approx -s/2*log(det(Kx+1/beta*eye(N))) - 1/(2*N)*trace((Kx+1/beta*eye(N))^(-1)*Ky )
% Input: 
%   Ky: N x N sym mat, Gram matrix for pairwise inner product for
%       Y = R^(N x s) with N samples, each row for each s dimensional
%                     sample
%   X : N x k, the matrix of latent representation, with N samples, each
%       row is a sample with k dimension
%  param
%      .N : # of samples
%      .s:  amient dimension
%      .k:  intrinsic dimension
%      .beta:  the noise varience, use to control the condition number of Kx
%  option
%      .kernelMethod:  
%          'linear' for linear kernel
%          'rbf' for Gaussian RBF kernel
%          'poly' for polynominial kernel
%       .kernelParm:
%           if 'linear', no need
%           elseif 'rbf', 
%               for variance sigma_k
%           elseif 'poly' 
%               for degree of polynominal k and bias term b
% 
% Output
%    Gm: N xk matrix. The natural gradient projected on the tangent space on the
%         manifold
%
% Written by Tianpei Xie, May 10 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N= param.N;
s= param.s;
k= param.k;

alpha = param.alpha;
beta = param.beta;
if strcmp(option.kernelMethod, 'linear')
  Kx = alpha*X*X'+ eye(N)/beta;
  Ge = -s*(Kx\X + Kx\(Ky/s*(Kx\X)));
elseif strcmp(option.kernelMethod, 'rbf')
  Ex = calckernel_v2(option, X);
  Kx = Ex+ eye(N)/beta;
  IKx = inv(Kx);
  sigma_k = option.kernelParm;
  S2 =  -s*Kx\Ex + Kx\(Ky*(Kx\Ex));
  S1 =  (-s*IKx+ IKx*Ky*IKx).*S2;
  Ge = 1/sigma_k*S1*X- 1/sigma_k*diag(diag(S2))*X;
  
end


