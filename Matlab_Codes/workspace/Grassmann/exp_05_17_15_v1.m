clear all;
close all;
clc
pathorgup = './dataset';
pathorgsub = '/Iris/';
pathorgfile = 'iris.mat';
load(strcat(pathorgup, pathorgsub,pathorgfile))

Y = iris(:,1:4);
label = iris(:,5);
idx11 = find(label==1);
idx12 = find(label==2);
idx13 = find(label==3);

Ky = Y*Y';
[N,s] = size(Y); 
k=2;


beta = 1;
if N<1000
   maxIter = 10*k*(N-k);
else
   maxIter = 250; %k*(N/4-k); 
end



param.N = N;
param.s = s;
param.k = k;
param.maxIter = maxIter;
param.beta = 1; %not good for too large
param.alpha = 1;
param.T = 100;
param.thresh = 1;
param.thres_norm = 1e-3;
param.tupper = 1e2; % 2e-1;
param.alphaT = 2e-4; %0.8*pi*beta*5e-4; %3e-1; %1; %1e-3;%5e-4;
param.dostop = 0;
option.kernelMethod = 'rbf';
option.kernelParm = 0.5;

[Us, Ss] = eig(Ky);
[dSs,Idx] = sort(diag(Ss),'descend');
Us = Us(:,Idx);
Ss = diag(dSs);

R = 1; %300;%10; %1; %300; %300;
Gnorm = zeros(1,maxIter+1);
diffXXnex = zeros(1,maxIter+1);
error = zeros(1,R);
Xarray = cell(1,R);
endidx = 0;
sel = 1;
minerror = inf;
minidx = R;
history_out = cell(1,R);
history_X0 = cell(1,R);
subspace_dist = zeros(1,maxIter+1);
subspace_hist = [];
countzero = 0;

%noise_orth = 1;

for r = 1:R
 
   Xu = randn(N,k);
   X0 = orth(Xu);
  %[X0, ~] = qr(Xu,0);
  %Vu = randn(k);
  %V0 = orth(Vu); %
  %X0 = Us(:,1:k)*Us(:,1:k)'*X0;+ noise_orth*(eye(N)-Us(:,1:k)*Us(:,1:k)')*X0;
 
history_X0{r} = X0;
[X, history, epsilon] = gplag(X0, Y, Ky, param, option, sel); %gplag_V(X0, Y, Ky, param, option, sel);
history_out{r} = history;
Xarray{r} = X;
%Proj = (eye(N)-X*X');
if strcmp(option.kernelMethod, 'linear')
 Rs = X'*Y;
   [UR,SR, VR] = svd(Rs);
   Xtst = X*UR;
   epsilon = k-norm(Xtst'*Us(:,1:k),'fro'); %k-trace(Xtst'*Us(:,1:k));
   error(r) = abs(epsilon); %real(trace(Proj*Ky*Proj));
else
    error(r) = abs(history.epsilon(history.iter));
end
 if (error(r)< minerror)
   minerror = error(r);
   minidx = r;
 end

% if (error(r)< 1e-2)
%     break;
% end

Gnorm =Gnorm+history.Gnorm;
endidx = max([endidx,history.iter]);
 if strcmp(option.kernelMethod, 'linear')
   if (error(r)< 1/beta)
    subspace_dist = subspace_dist+ history.epsilon;
    countzero= countzero+1;
   end
 else
    diffXXnex = diffXXnex+ history.epsilon; 
 end
end
Gnorm=Gnorm/R;
subspace_dist = subspace_dist/countzero;
 if strcmp(option.kernelMethod, 'rbf')
   diffXXnex = diffXXnex/R;
 end

figure(1)
plot(1:endidx, Gnorm(1:endidx),'Linewidth',2)
xlabel('iteration','Fontsize',18)
ylabel('$\|\mathbf{G}_{t}\|/\sqrt{kN}$','Interpreter','LaTex','Fontsize',18);
 
if strcmp(option.kernelMethod, 'rbf')
 figure(2)
 plot(1:endidx, diffXXnex(1:endidx),'Linewidth',2)
 xlabel('iteration','Fontsize',18)
 ylabel('$k-\|\mathbf{X}^{T}_{t}\mathbf{X}_{t-1}\|$','Interpreter','LaTex','Fontsize',18);
end
figure(8)
plot(X(idx11,1)',X(idx11,2)', 'ob');
hold on;
plot(X(idx12,1)',X(idx12,2)', 'xr');
plot(X(idx13,1)',X(idx13,2)', '+g');
hold off
xlabel('first direction','Fontsize',18)
ylabel('second direction','Fontsize',18);

figure(7)
[coeff, score, Upca] = pca(Y); 
Xpca = Y*Upca(:,1:k);
plot(Xpca(idx11,1)',Xpca(idx11,2)', 'ob');
hold on;
plot(Xpca(idx12,1)',Xpca(idx12,2)', 'xr');
plot(Xpca(idx13,1)',Xpca(idx13,2)', '+g');
hold off
xlabel('first PCA direction','Fontsize',18)
ylabel('second PCA direction','Fontsize',18);


% if sel == 1
% title(sprintf('Gradient descent on Grassmannian for dimensionality reduction (N= %d, k=%d, s=%d)',N,k,s))
% else
% title(sprintf('Conjugate Gradient descent on Grassmannian for dimensionality reduction (N= %d, k=%d, s=%d)',N,k,s))    
% end
% for r=1:R
% history_opt = history_out{r};
% figure(3)
% plot(1:history_opt.iter, history_opt.Gnorm(1:history_opt.iter),'Linewidth',2)
%  xlabel('iteration(in-loop)','Fontsize',14)
%  ylabel('$\|\mathbf{G}_{t}\|/\sqrt{kN}$','Interpreter','LaTex','Fontsize',14);
% if sel == 1
%     title(sprintf('r= %d ',r))
% %title(sprintf('Gradient descent on Grassmannian for linear dimensionality reduction (N= %d, k=%d, s=%d)',N,k,s))
% else
% title(sprintf('Conjugate Gradient descent on Grassmannian for linear dimensionality reduction (N= %d, k=%d, s=%d)',N,k,s))    
% end
% drawnow; pause(.5)
% end

% figure(7);
% history_opt = history_out{10};
% str = cell(1,k);
% color = [{'r'},{'b'}];
% for i=1:k
%    plot(1:maxIter+1, history_opt.UUSigma(i,:),color{i});
%    hold on;
%    str{i} = strcat('\fontsize{18}{0}\selectfont$\cos(\phi', '_', num2str(i) ,')$');
% end
% hold off;
% h =legend(str);
% set(h,'Interpreter','LaTex','Fontsize',18)
% xlabel('Iteration','Fontsize',18);
% ylabel('\fontsize{18}{0}\selectfont$\cos(\phi)$','Interpreter','LaTex','Fontsize',18);
% 
% for r=1:5:R
%  history_opt = history_out{r};
%  figure(6) 
% for i=1:k
%    plot(1:maxIter+1, history_opt.sv(i,:));
%    hold on;
% end
% hold off;
% title(sprintf('r= %d ',r))
% drawnow; pause(.5)
% 
%  figure(3)
%  plot(1:history_opt.iter, history_opt.epsilon(1:history_opt.iter),'Linewidth',2)
%   xlabel('iteration ','Fontsize',14)
%   ylabel('subspace distance')
%   %ylabel('$\|\mathbf{G}_{t}\|/\sqrt{kN}$','Interpreter','LaTex','Fontsize',14);
% title(sprintf('r= %d ',r))
% drawnow; pause(.5)
% end
% 
% figure(4)
% plot(1:endidx, subspace_dist(1:endidx),'Linewidth',2)
% xlabel('iteration','Fontsize',18)
% ylabel('subspace-distance','Fontsize',18);

% if sel == 1
% title(sprintf('Gradient descent on Grassmannian for dimensionality reduction (N= %d, k=%d, s=%d)',N,k,s))
% else
% title(sprintf('Conjugate Gradient descent on Grassmannian for dimensionality reduction (N= %d, k=%d, s=%d)',N,k,s))    
% end
%  figure(2)
%  %semilogy(1:R, error,'Linewidth',2)
%  plot(1:R, error,'Linewidth',2)
%  xlabel('iteration(out-loop)','Fontsize',14)
%  ylabel('error','Fontsize',14);
%  %ylabel('log(error)','Fontsize',14);
%  title(sprintf('Reconstruction error for learned subspace'))
 
%  figure(5)
%  hist(error)
%  Nhist = hist(error);
%  xlabel('subspace-distance','Fontsize',18);
%  ylabel('unnormalized frquency ','Fontsize',18)
%  axis([0,k,0,R]);
%  
%  save(strcat('exp_',num2str(today),'.mat'), 'N','k', 's','sigma', ...
%      'Xtr', 'W', 'Ky', 'Y', 'beta', 'maxIter', 'error', 'subspace_hist','subspace_dist', 'Nhist',...
%      'option','param','history_out','history_X0', 'Xarray', 'minidx' );
 
 