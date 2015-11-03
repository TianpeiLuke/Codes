function Eta= opt_size(Ky, X, U, k)
p = zeros(1,k);
r = zeros(1,k);
Eta = zeros(k);
[N,k] = size(X);
Z = eye(N)- X*X';
for ii= 1:k 
    
  p(ii) = sqrt(trace(X'*Ky*X));
  r(ii) = sqrt(trace(Z'*Ky*Z));
   Eta(ii,ii) = atan(r(ii)/p(ii));
end


end


