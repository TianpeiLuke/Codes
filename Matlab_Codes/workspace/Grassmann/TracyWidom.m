function [Fs fs] = TracyWidom(beta,s)
%--------------------------------------------------------------------------
% Syntax:       [Fs fs] = TracyWidom(beta,s);
%               
% Inputs:       beta = {1,2,4} is the Hermite class parameter
%               
%               s is an array of sample points
%               
% Outputs:      Fs is an array of same size as s containing the values of
%               the Tracy Widom CDF evaluated at the values in s
%               
%               Fs is an array of same size as s containing the values of
%               the Tracy Widom PDF evaluated at the values in s
%               
% Description:  This function evaluates the Tracy Widom CDF and PDF with
%               parameter beta at the points in s
%               
% Reference:    Chapter 9 of
%               "A. Edelman, R. Nadakuditi, "Random Matrix Theory".
%               http://web.eecs.umich.edu/~rajnrao/Acta05rmt.pdf
%               
% Author:       Brian Moore
%               brimoor@umich.edu
%               
% Date:         May 18, 2015
%--------------------------------------------------------------------------

% Solve Painleve II equation numerically on a dense grid
deq       = inline('[y(2); s * y(1) + 2 * y(1)^3]','s','y');
sspan     = linspace(8,-8,10000);
y0        = [airy(8); airy(1,8)];
[sgrid y] = ode45(deq,sspan,y0,odeset('reltol',1e-13,'abstol',1e-15));
q         = y(:,1);

% Numerical integration
dI0 = 0;     % Initial value
I0  = 0;     % Initial value
J0  = 0;     % Initial value
ds = diff(sgrid);
dI  = -[0; cumsum(0.5 * ( q(1:(end-1)).^2 + q(2:end).^2) .* ds)] + dI0;
I   = -[0; cumsum(0.5 * (dI(1:(end-1))    + dI(2:end))   .* ds)] + I0;
J   = -[0; cumsum(0.5 * ( q(1:(end-1))    +  q(2:end))   .* ds)] + J0;

% Compute Fs and fs on dense grid
F2 = exp(-I);
if (beta == 1)
    Fs = sqrt(F2 .* exp(-J));
    fs = gradient(Fs,sgrid);
elseif (beta == 2)
    Fs = F2;
    fs = gradient(Fs,sgrid);
elseif (beta == 4)
    Fs = 0.5 * sqrt(F2) .* (exp(0.5 * J) + exp(-0.5 * J));
    fs = gradient(Fs,sgrid / (2^(2 / 3)));
else
    error('beta not supported');
end

% Interpolate to compute values at actual sample points
Fs = interp1(sgrid,Fs,s);
fs = interp1(sgrid,fs,s);
