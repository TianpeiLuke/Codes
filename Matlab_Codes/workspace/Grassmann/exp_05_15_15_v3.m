clear all;
close all;
clc
N = 100;%2000;%2000;  %1000;
k = 1; %2;
s = 20; %if s=2? not stable?
sigma = 1;

W = 1e-1*randn(k,s);
Xtr = randn(N,k);


beta = 10;
maxIter = k*(N-k);

%Y = Xtr*W/sqrt(N*k)+ 1/beta*randn(N,s);
Xtr = orth(Xtr);
Y = Xtr*W;



%+ 1/beta*randn(N,s);
Noise = randn(N);
%Noise = (eye(N)- Xtr*Xtr')*randn(N,s);
%Noise = (eye(N)- Xtr*Xtr')*randn(N);
%Y = Xtr*W + (1/beta)*Noise;
%Noise = Xtr*Xtr'*randn(N);
Ky = Y*Y'+ (1/beta)*(Noise*Noise');

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
param.alphaT = 1e-3; %0.8*pi*beta*5e-4; %3e-1; %1; %1e-3;%5e-4;
param.dostop = 0;
option.kernelMethod = 'linear';

[Us, Ss] = eig(Ky);
[dSs,Idx] = sort(diag(Ss),'descend');
Us = Us(:,Idx);
Ss = diag(dSs);

R = 10;%10;%300;%10; %1; %300; %300;
Gnorm = zeros(1,maxIter+1);
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

noise_orth = 1;

for r = 1:R
 if r<=R   
   Xu = randn(N,k);
   X0 = orth(Xu);
  %[X0, ~] = qr(Xu,0);
  %Vu = randn(k);
  %V0 = orth(Vu); %
  X0 = Us(:,1:k)*Us(:,1:k)'*X0;+ noise_orth*(eye(N)-Us(:,1:k)*Us(:,1:k)')*X0;
 elseif r>R
   Proj = (eye(N)-X*X');
   [Xdelta, Sdelta] = eig(Proj);
   [dSs,IdD] = sort(diag(real(Sdelta)),'descend');
   Xdelta = Xdelta(:,IdD);
   epsilon2 = 1e-1*randn(1); %1/(100*R); %1e-1*randn(1);
   X0 = (X+epsilon2*Xdelta(:,[1:k]))/(sqrt(1+epsilon2^2));
 end
%X0 = Us(:,[k+1:2*k]);
history_X0{r} = X0;
[X, history, epsilon] = gplag_optstepsize(X0, Y, Ky, param, option, sel); %gplag(X0, Y, Ky, param, option, sel);
history_out{r} = history;
Xarray{r} = X;
%Proj = (eye(N)-X*X');
 Rs = X'*Y;
   [UR,SR, VR] = svd(Rs);
   Xtst = X*UR;
   epsilon = k-norm(Xtst'*Us(:,1:k),'fro'); %k-trace(Xtst'*Us(:,1:k));
error(r) = abs(epsilon); %real(trace(Proj*Ky*Proj));
if (error(r)< minerror)
  minerror = error(r);
  minidx = r;
end
% if (error(r)< 1e-2)
%     break;
% end
Gnorm =Gnorm+history.Gnorm;
endidx = max([endidx,history.iter]);
if (error(r)< 1/beta)
subspace_dist = subspace_dist+ history.epsilon;
countzero= countzero+1;
end

end
Gnorm=Gnorm/R;
subspace_dist = subspace_dist/countzero;

% figure(1)
% plot(1:endidx, Gnorm(1:endidx),'Linewidth',2)
% xlabel('iteration','Fontsize',18)
% ylabel('$\|\mathbf{G}_{t}\|/\sqrt{kN}$','Interpreter','LaTex','Fontsize',18);

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


for r=1:R
 history_opt = history_out{r};
 figure(6) 
for i=1:k
   plot(1:maxIter+1, history_opt.sv(i,:));
   hold on;
end
hold off;
title(sprintf('r= %d ',r))
drawnow; pause(.5)

 figure(3)
 plot(1:history_opt.iter, history_opt.epsilon(1:history_opt.iter),'Linewidth',2)
  xlabel('iteration ','Fontsize',14)
  ylabel('subspace distance')
  %ylabel('$\|\mathbf{G}_{t}\|/\sqrt{kN}$','Interpreter','LaTex','Fontsize',14);
title(sprintf('r= %d ',r))
drawnow; pause(.5)
end
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
 
 figure(5)
 hist(error)
 Nhist = hist(error);
 xlabel('subspace-distance','Fontsize',18);
 ylabel('unnormalized frquency ','Fontsize',18)
 axis([0,k,0,R]);
 
 save(strcat('exp_',num2str(today),'.mat'), 'N','k', 's','sigma', ...
     'Xtr', 'W', 'Ky', 'Y', 'beta', 'maxIter', 'error', 'subspace_hist','subspace_dist', 'Nhist',...
     'option','param','history_out','history_X0', 'Xarray', 'minidx' );
 
 