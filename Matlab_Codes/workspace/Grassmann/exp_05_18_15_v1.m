clear all;
close all;
clc
N = 100; %100;%2000;%2000;  %1000;
k = 2; %2;
s = 20; %if s=2? not stable?
sigma = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  The following parameters are important:
%1. beta: [0,1, 10]
%2. alphaT: control initial stepsize => large initial step will converge to
%           local optima fast; small initial step will use a lot of
%           iterations
%3. alpharr: (0,1], the speed of diminishing of stepsize => avoid some trival local
%            optima for increases it: a large alpharr results in large
%            maxIter
%4. maxIter: depends on alpharr more than alphaT; in order of k(N-k)
% 
%5. initial subspace X0: if X0 and Xopt is close, then we can find the optimal solution  
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

beta = 10;
if N<1000
   maxIter = 2*k*(N-k);
else
   maxIter = 250; %k*(N/4-k); 
end
%% data generation

W = 1e-1*randn(k,s);
Xtr = randn(N,k);
Xtr = normc(Xtr);

Noise = randn(N,s)/sqrt(N);

Y = Xtr*W+ 1/sqrt(beta)*Noise;
Y = bsxfun(@minus,Y,mean(Y));  %centered data
 
Ky = Y*Y'/s;

%% parameter setting
param.N = N;
param.s = s;
param.k = k;
param.maxIter = maxIter;
param.beta = 10; %not good for too large
param.alpha = 1e3;
param.T = 100;
param.thresh = 1;
param.thres_norm = 1e-3;
param.tupper = 1e2; 
param.alphaT = 2e-1;
param.alpharr = 1/4;
param.dostop = 0;
param.gstep = 0;%1e-3; %1;
param.lambda = 100;
param.g0const = 1;


option.kernelMethod = 'linear';

% [Us, Ss] = eig(Ky);
% [dSs,Idx] = sort(diag(Ss),'descend');
% Us = Us(:,Idx);
% Ss = diag(dSs);

[Us, Ds, Vs]=svd(Y);
dSs = diag(Ds*Ds'/s);


% if strcmp(option.kernelMethod, 'linear')
%    betainv = 1/(sum(dSs(s+1:end))/(s*(N-s))); 
% end



R = 10;%10; %1; %300; %300;
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

Inisub_dist_opt = zeros(2,R);
Inisub_dist_prj = zeros(2,R);

noise_orth = 0; %1e-2;
dt=0;
param.dt= dt;
for r = 1:R
  
   Xu = randn(N,k);
   [X0, G0, ~] = svd(Xu,'econ');
   X0 = orth(Xu);
  %[X0, ~] = qr(Xu,0);
  %Vu = randn(k);
  %V0 = orth(Vu); 
   %Rs = X0'*Y;
   %[UR,SR, VR] = svd(Rs);
   %X0 = X0*UR;
   idxdiff = setdiff([1:N],[1+dt:k+dt]);
   Inisub_dist_opt(1,r) = k- norm(X0'*Us(:,1+dt:k+dt),'fro'); 
   Inisub_dist_opt(2,r) = k- norm(X0'*Us(:,idxdiff),'fro'); 
   
%   X0 = Us(:,1:k)*Us(:,1:k)'*X0 + ...
%   noise_orth*(eye(N)-Us(:,1:k)*Us(:,1:k)')*X0;   %important?
   X0 = Us(:,1+dt:k+dt)*Us(:,1+dt:k+dt)'*X0 + ...
   noise_orth*(eye(N)-Us(:,1+dt:k+dt)*Us(:,1+dt:k+dt)')*X0;   %important?

   Inisub_dist_prj(1,r) = k- norm(X0'*Us(:,1+dt:k+dt),'fro'); 
   Inisub_dist_prj(2,r) = k- norm(X0'*Us(:,idxdiff),'fro');
%X0 = Us(:,[k+1:2*k]);
   history_X0{r} = X0;
  

[X, Gamma, history, epsilon] = gplag_weight(X0, G0, Y, Ky, param, option, sel); 
history_out{r} = history;
Xarray{r} = X;
%Proj = (eye(N)-X*X');
 Rs = X'*Y;
   [UR,SR, VR] = svd(Rs);
   Xtst = X*UR;
   epsilon = k-norm(Xtst'*Us(:,1+dt:k+dt),'fro'); %k-trace(Xtst'*Us(:,1:k));
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

figure(7);
history_opt = history_out{10};
str = cell(1,k);
color = [{'r'},{'b'}];
for i=1:k
   plot(1:maxIter+1, history_opt.UUSigma(i,:),color{i});
   hold on;
   str{i} = strcat('\fontsize{14}{0}\selectfont$\cos(\phi', '_', num2str(i) ,')$');
end
hold off;
h =legend(str);
set(h,'Interpreter','LaTex','Fontsize',18)
xlabel('Iteration','Fontsize',18);
ylabel('\fontsize{14}{0}\selectfont$\cos(\phi)$','Interpreter','LaTex','Fontsize',14);
axis([0, maxIter+1, 0, 1])
drawnow; pause(.5)

figure(9);
plot(Inisub_dist_opt(1,:), Inisub_dist_opt(2,:),'xr');
%xlabel('runs','Fontsize',18)
%ylabel('Subspace distance to opt','Fontsize',18)
xlabel('distance to optimal');
ylabel('distance to orthogonal-optimal');
title('Initial subspace distance','Fontsize',16)
grid on
drawnow; pause(.5)

figure(8);
plot(Inisub_dist_prj(1,:), Inisub_dist_prj(2,:),'or');
%xlabel('runs','Fontsize',18)
%ylabel('Subspace distance to opt','Fontsize',18)
xlabel('distance to optimal');
ylabel('distance to orthogonal-optimal');
title('Initial subspace distance','Fontsize',16)
grid on
drawnow; pause(.5)

figure(10)
stem(history_opt.gamma(1,:), 'b');
hold on;
stem(history_opt.gamma(2,:), 'r');
hold off;
xlabel('iteration','Fontsize',18);
ylabel('\fontsize{16}{0}\selectfont$\Gamma$', 'Interpreter','LaTex','Fontsize',16)
h =legend('\fontsize{18}{0}\selectfont$\Gamma_{11}$','\fontsize{18}{0}\selectfont$\Gamma_{22}$');
set(h, 'Interpreter','LaTex','Fontsize',18);


figure(11)
stem(history_opt.Ggnorm)
xlabel('iteration','Fontsize',16);
ylabel('\fontsize{16}{0}\selectfont$\|\Gamma\|_{F}$', 'Interpreter','LaTex','Fontsize',16)

figure(12)
plot(history_opt.objval,'Linewidth',2)
xlabel('iteration','Fontsize',18);
ylabel('objective value','Fontsize',18)
title('objvalue')
drawnow; pause(.5)

optvalues = zeros(1,R+1);
for r=1:R
    perm0 = randperm(s);
    Xtt= Us(:,perm0(1:k));
    Gamma = eye(k);
    optvalues(r) = neglogP(Ky, Xtt, Gamma, param, option);
end
Xtt= Us(:,1:k);
optvalues(R+1) = neglogP(Ky, Xtt, Gamma, param, option);
plot(1:R+1, optvalues);
drawnow; pause(.5)


for r=10 %1:5:R
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
 
 