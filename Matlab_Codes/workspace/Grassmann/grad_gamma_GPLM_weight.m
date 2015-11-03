function gd_gamma = grad_gamma_GPLM_weight(Ky, X, gamma, param, option)
k= param.k;
beta = param.beta;
alpha = param.alpha;


if strcmp(option.kernelMethod, 'linear')
   md = alpha*diag(X'*Ky*X);
   pweight = beta./(1+ beta*gamma.^2);
   pb = (beta*gamma.^2)./(1+ beta*gamma.^2);
   
   gd_gamma_pre = pweight.*(1 - pweight.*md);
   gd_gamma = gamma.*gd_gamma_pre; %- param.lambda*gamma;
elseif strcmp(option.kernelMethod, 'rbf')
    gd_gamma = zeros(k,1);
end