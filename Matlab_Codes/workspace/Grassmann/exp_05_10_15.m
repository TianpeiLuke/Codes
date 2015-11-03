N = 100;  %1000;
k = 2;
s = 10;
sigma = 1;

W = randn(k,s);
X = sigma*randn(N,k);
beta = 1;
maxIter = k*(N-k);

Y = X*W + 1/beta*randn(N,s);

Ky = Y*Y';

param.N = N;
param.s = s;
param.k = k;
param.maxIter = maxIter;
param.beta = 1;
param.alpha = 1;
param.T = 100;
param.thresh = 2e-2;
param.thres_norm = 1e-1;
param.tupper = 1e2; % 2e-1;
option.kernelMethod = 'linear';

[Us, Ss] = eig(Ky);
[dSs,Idx] = sort(diag(Ss),'descend');
Us = Us(:,Idx);
Ss = diag(dSs);
%Xu = X + sigma*1e-2*randn(N,k);
R = 150;
Gnorm = zeros(1,maxIter+1);
endidx = 0;
sel = 1;
for r = 1:R
Xu = randn(N,k);
[X0, ~] = qr(Xu,0);%%%
%X0 = Us(:,[k+1:2*k]);

[X, history, epsilon] = conjugate_grad(X0, Ky, param, option, sel);
Gnorm =Gnorm+history.Gnorm;
endidx = max([endidx,history.iter]);
end
Gnorm=Gnorm/R;
figure(1)
plot(1:endidx, Gnorm(1:endidx),'Linewidth',2)
xlabel('iteration','Fontsize',14)
ylabel('$\|\mathbf{G}_{t}\|/\sqrt{kN}$','Interpreter','LaTex','Fontsize',14);
if sel == 1
title(sprintf('Gradient descent on Grassmannian for linear dimensionality reduction (N= %d, k=%d, s=%d)',N,k,s))
else
title(sprintf('Conjugate Gradient descent on Grassmannian for linear dimensionality reduction (N= %d, k=%d, s=%d)',N,k,s))    
end