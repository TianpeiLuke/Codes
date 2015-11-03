function [Gm] = grad_man_GPLM_weight(Ky, X, weight, param, option)
%% compute the natrual gradient of log p(Y|X) at X along the geodesic
% log p(Y|X) \approx -s/2*log(det(Kx+1/beta*eye(N))) - 1/(2*N)*trace((Kx+1/beta*eye(N))^(-1)*Ky )
% Input: 
%   Ky: N x N sym mat, Gram matrix for pairwise inner product for
%       Y = R^(N x s) with N samples, each row for each s dimensional
%                     sample
%   Kx: N x N sym mat, Gram (Kernel) matrix for pariwise inner product for
%       X;  
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
beta = param.beta;
Proj = proj_comp(X, N);
if strcmp(option.kernelMethod, 'linear')
  Ge = grad_comp_GPLM_weight(Ky, X, weight, param, option);
  Gm = Proj*Ge;
%     d = (weight.^2)./(1/beta+ weight.^2);
%     D = diag(d);
%     Gm = beta*Proj*Ky*X*D;
elseif strcmp(option.kernelMethod, 'rbf')
  Ge = grad_comp_GPLM_weight(Ky, X, weight, param, option);
  Gm = Proj*Ge;     
end











