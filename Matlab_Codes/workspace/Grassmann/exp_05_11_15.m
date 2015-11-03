


N = 1000;
k = 2;
s = 10;
sigma = 1;

W = randn(k,s);
X = sigma*randn(N,k);
beta = 100;
maxIter = k*(N-k);

Y = X*W + 1/beta*randn(N,s);

Ky = Y*Y';

param.N = N;
param.s = s;
param.k = k;
param.maxIter = maxIter;
param.beta = beta;

Xu = randn(N,k);
[X0, ~] = qr(Xu,0);

option.kernelMethod = 'linear';
Gm0 = grad_man_GPLM(Ky, X0, param, option);

H = -Gm0;
G=  Gm0;
G_pre = Gm0;
[U,S,V] = svd(H,0);
X = X0;

tmin = 0.01
Xt=  X*V*diag(cos(tmin*diag(S)))*V' + U*diag(sin(tmin*diag(S)))*V';
Xt'*Xt