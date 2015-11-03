function P = proj_comp(X, N)

P = eye(N) - X*X';