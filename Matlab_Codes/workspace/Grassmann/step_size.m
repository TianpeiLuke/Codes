function [tmin]= step_size(X, Ky, matparm, param, option, history)
N= param.N;
s= param.s;
k= param.k;
T= param.T;
tupper = param.tupper;
beta = param.beta;
pdt = param.pdt;
ti = param.time;

U = matparm.U{1};
S = matparm.S{1}; 
V = matparm.V{1}; 

%T = 200;
%dt = pi/(2*max(abs(diag(S))));
%t = [1:T]*dt; 

t= linspace(1e-5, tupper, T);
L = zeros(1,T);
for i=1:length(t)
%   if i==ceil(length(t)/2)
%    display(sprintf('#i:%d',i));
%   end
 X_temp = X*V*diag(cos(t(i)*diag(S)))*V' + U*diag(sin(t(i)*diag(S)))*V'; 
 if strcmp(option.kernelMethod, 'linear')
    %K = X_temp*X_temp' + 1/(beta)*eye(N); 
    L(i) = - beta/2*trace((eye(N) - beta/(1+beta)*X_temp*X_temp')*Ky); 
 elseif strcmp(option.kernelMethod, 'rbf')  
    K = calckernel_v2(option,X_temp)+ 1/(beta)*eye(N);
    L(i) = -s/2*log(det(K))- 1/2*trace(K\Ky);
 end
end
[~,idxMin] = min(L);
history.tl{ti} = L;
fl= abs(fft(L));
history.fl{ti} = fl;
tmin = t(idxMin);
display(sprintf('tmin: %f',tmin));
end