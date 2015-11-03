
% [ K, D2, NN, [D2full] ] = graphDistKernel( X, D2, NN, idxSet, param, [D2full] )
% computes a kernel matrix 'K' based on graph distances.
%
% INPUT / OUTPUT:
%   X       double[d*m]  'm' data points in Euclidean 'd'-space
%   D2     double[nn*m]  distances of 'm' data points to 'nn' respective nearest neighbors
%   NN      int32[nn*m]  indices of 'nn' nearest neighbors for each of 'm' data points
%   idxSet  double[1*l]  indices of subset for which kernel matrix rows will be computed
%   D2full  double[m*m]  full matrix of pairwise distances
%   K       double[m*l]  resulting kernel matrix
%   param        struct of parameters
%     .pathNorm  norm used for path distance computation
%                NOTE: distances will be normalized before
%     .nofNn     number of nearest neighbors; 0 for full graph
%     .annEps    allowed approximation error for approx. nearest neighbor;
%                if <0 then the full distance matrix will be sorted
%     .rbf       name of rbf: 'gauss', 'laplace'
%     .sigma     kernel width; +inf for matrix of squared path distances
%                NOTE: taken relative to "default" sigma
% 
% USAGE:
%   graphDistKernel( X,  D2, NN, idxSet, param )
%   graphDistKernel( [], D2, NN, idxSet, param )
%     'NN' and 'D2' will be used as given.
%   graphDistKernel( X, [], [], idxSet, param )
%     'NN' and 'D2' will be computed from 'X'.
%   graphDistKernel( [], [], [], idxSet, param, D2full )
%     'NN' and 'D2' will be computed from 'D2full', if 'param.annEps' < 0.
% 
% (C) A. Zien and O. Chapelle, MPI for biol. Cybernetics, Germany


function [ K, D2, NN, D2full ] = graphDistKernelC( X, D2, NN, idxSet, param, D2full );

% ==== check input
assert( 5 <= nargin & nargin <= 6 );
assert( 0 <= param.pathNorm );
assert( 0 <= param.nofNn );
assert( 0 < param.sigma && param.sigma <= +inf );
if( ~ exist( 'D2full', 'var' ) )
  D2full = [];
end;
% --- check NN and D2, if supplied
if( length(D2) ~= 0 )
  if( param.nofNn == 0 )
    [ m, m_ ] = size( D2 );
    assert( m == m_ );
    assert( length(NN) == 0 );
  else
    [ nofNn, m ] = size( D2 );
    assert( nofNn == param.nofNn );
    assert( size(NN,1) == nofNn );
    assert( size(NN,2) == m );
  end;
  if( length(X) ~= 0 )
    assert( size(X,2) == m );
  end;
end;

% === compute nearest neighbors (NN) and squared distances (D2)
if( length(D2) == 0 )
  assert( length(NN) == 0 );
  [ NN, D2 ] = calcNnDists( X, D2full, param.nofNn, param.annEps );
end;

% === compute matrix of squared path distances (E2)
meanD2 = mean( sqrt( D2(:) ) ) ^ 2;
if( param.pathNorm == 0 & param.nofNn == 0 & 0 )
  % - for pathNorm==0, the direct connection is always shortest
  E2 = D2( :, idxSet ) / meanD2;
else
  E2 = rhoPathDists2( D2/meanD2, NN, idxSet, param.pathNorm );
end;
% --- return path distance matrix?
if( param.sigma == +inf )
  K = E2;
  %%% % --- symmetrize distances
  %%% if( param.nofNn ~= 0 )
  %%%   %K(idxSet,:) = min( K(idxSet,:), K(idxSet,:)' );
  %%%   K(idxSet,:) = ( K(idxSet,:) + K(idxSet,:)' ) ./ 2;
  %%% end;
  return;
end;

% === choose default kernel width (sigma)
k = length( param.ks );
sigma = param.sigma * calcDefaultSigma( E2(:), k );

% === compute kernel matrix (K)
mustSymmetrize = param.nofNn ~= 0;
K = calcRbfKernel( E2, param.rbf, sigma, mustSymmetrize );


