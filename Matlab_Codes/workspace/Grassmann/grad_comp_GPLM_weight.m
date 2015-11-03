function [Ge] = grad_comp_GPLM_weight(Ky, X, weight, param, option)
%% compute the Euclidean gradient of log p(Y|X) at X
% log p(Y|X) \approx -s/2*log(det(Kx+1/beta*eye(N))) - 1/(2*N)*trace((Kx+1/beta*eye(N))^(-1)*Ky )
% Input: 
%   Ky: N x N sym mat, Gram matrix for pairwise inner product for
%       Y = R^(N x s) with N samples, each row for each s dimensional
%                     sample
%   X : N x k, the matrix of latent representation, with N samples, each
%       row is a sample with k dimension
% weight: k x 1, the scale vector for k-basis
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
beta = param.beta;
if strcmp(option.kernelMethod, 'linear')
    Gamma = diag(weight);
    Kx =X*(Gamma.^2)*X'+ eye(N)/beta;
   d = (weight.^2)./(1/beta+ weight.^2);
   D = diag(d);
   Ge = X*D - (Kx\Ky)*X*D; 
   Ge = -Ge; 
   %Ge = -s*(Kx\X + Kx\(Ky/s*(Kx\X)))*D;
  %Ge = *Ge;
elseif strcmp(option.kernelMethod, 'rbf')
  d = weight.^2;
  D = diag(d); %add weight
  Gamma = diag(weight);
  Ex = calckernel_v2(option, X*Gamma); %add weight
  Kx = Ex+ eye(N)/beta;
  IKx = inv(Kx);
  sigma_k = option.kernelParm;
  S2 =  Kx\Ex - Kx\(Ky*(Kx\Ex));
  S1 =  (IKx- Kx\(Ky*IKx)).*S2;
  Ge = 1/sigma_k*S1*X*D- 1/sigma_k*diag(diag(S2))*X*D;
  Ge = -Ge;
end


