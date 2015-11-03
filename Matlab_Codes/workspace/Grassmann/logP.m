function objval = logP(Ky, X, param, option)

N= param.N;
s= param.s;
beta = param.beta;

if strcmp(option.kernelMethod, 'linear')
    K = X*X' + 1/(beta)*eye(N); 
    objval =  -1/2*trace(K\Ky); 
 elseif strcmp(option.kernelMethod, 'rbf')  
    K = calckernel_v2(option,X)+ 1/(beta)*eye(N);
    objval = -s/2*log(det(K))- 1/2*trace(K\Ky);
 end