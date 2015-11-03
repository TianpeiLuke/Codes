function [h x p] = histn(y,varargin)
%--------------------------------------------------------------------------
% Syntax:       histn(y);
%               histn(y,Nbins);
%               histn(y,x);
%               h = histn(y);
%               h = histn(y,Nbins);
%               h = histn(y,x);
%               
% Inputs:       y is a vector of data (say N samples)
%               
%               Nbins is the number of bins to use, of form:
%               
%                       linspace(min(y),max(y),Nbins)
%               
%               x is a vector of bin centers. If x and Nbins are not
%               specified, the default bins are:
%               
%               linspace(min(y),max(y),ceil(N / DefaultPtsPerPDFBin))
%               
% Outputs:      h is the handle to the normalized bar plot
%               
% Description:  This script plots the normalized histogram of the data
%               vector y using the bin centers in x
%               
% Author:       Brian Moore
%               brimoor@umich.edu
%               
% Date:         October 7, 2012
%--------------------------------------------------------------------------

% Constants
DefaultMinBins = 2;
DefaultPtsPerBin = 25;
width = 0.6;

y = y(:)';
y(isnan(y)) = [];
if (nargin == 2)
    if (length(varargin{1}) == 1)
        Nbins = varargin{1};
        x = linspace(min(y),max(y),Nbins);
    else
        x = varargin{1};
        x = x(:)';
    end
else
    N = length(y);
    x = linspace(min(y),max(y),max(DefaultMinBins,ceil(N / DefaultPtsPerBin)));
end
n = length(x);

edges = (x(1:(end-1)) + x(2:end))/2;

p = zeros(1,n);
p(1) = sum(y <= edges(1));
for i = 2:(n-1)
    p(i) = sum((y > edges(i-1)) .* (y <= edges(i)));
end
p(n) = sum(y > edges(n-1));
p = p / sum([(edges(2) - edges(1)),(edges(2:end) - edges(1:(end-1))),(edges(end) - edges(end-1))] .* p);

h = bar(x,p,width);
title('Normalized Histogram of Sample Data');
grid on;
