function objval = neglogP(Ky, X, Gamma, param, option)

N= param.N;
s= param.s;
beta = param.beta;

if strcmp(option.kernelMethod, 'linear')
    K = X*Gamma.^2*X' + 1/(beta)*eye(N); 
    objval =  1/2*log(det(K))+ 1/2*trace(K\Ky); 
 elseif strcmp(option.kernelMethod, 'rbf')  
    K = calckernel_v2(option,X*Gamma)+ 1/(beta)*eye(N);
    objval = 1/2*log(det(K))+ 1/2*trace(K\Ky);
 end