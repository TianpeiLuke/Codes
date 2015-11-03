%
% Verify Theorem 9.2 of http://web.eecs.umich.edu/~rajnrao/Acta05rmt.pdf
%
%
% Brian Moore
% brimoor@umich.edu
% May 18, 2015
%

% Knobs
m = 200; % Must have m >= n
n = 150;
Ntrials = 10000;

% Perform simulations
lambda1 = nan(1,Ntrials);
for i = 1:Ntrials
    G = randn(m,n);
    W1 = G' * G;
    lambda1(i) = max(eig(W1));
end

% Scale eigenvalues
mu = (sqrt(m - 1) + sqrt(n))^2;
sigma = (sqrt(m - 1) + sqrt(n)) * (1 / sqrt(m - 1) + 1 / n)^(1 / 3);
lambda1p = (lambda1 - mu) / sigma;

% Compute Tracy-Widom distribution
s = linspace(-5,5,1000);
[Fs fs] = TracyWidom(1,s);

% Plot results
figure;
histn(lambda1p,50);
hold on;
plot(s,fs,'r');
title('(\lambda_{max}(W_1(m,n)) - \mu) / \sigma');
