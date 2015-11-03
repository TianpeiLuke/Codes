function [LapN,LapUN, SpC, CKSym, CAbs]= SSC_partial(Y, affine,alpha, rho , K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  run Partial SSC where the sparse code is generated and the Laplacian is
%    returned
%
% Input: 
%     Y: Dx1 cell array, each consist of d x N data matrix with N samples 
%   affine: if affine constrain is added
%   alpha:  regularization parameter of sparse coding
%
% Output:     
%   LapN: Dx1 cell array, each for a N x N Laplacian graph 
%   SpC:  Dx1 cell array, each for a N x N sparse code matrix 
% CKSym:  Dx1 cell array, each for a N x N sparse affinity matrix
%  CAbs:  Dx1 cell array, each for a N x N absolute value of sparse code
%
%   
%  written by Tianpei Xie, Mar_27_2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if (nargin < 2)
    % default subspaces are linear
    affine = false; 
end
if (nargin < 3)
    % default regularizarion parameters
     alpha = 20;
end
if (nargin < 4)
    rho = 1;
end 
if (nargin < 5)
    K = 0;
end 

D = length(Y);
[d,N] = size(Y{1});

 SpC = cell(D,1);
  
 for ii=1:D  
     SpC{ii} = admmLasso_mat_func(Y{ii}, affine, alpha, 2*10^-4, 400);
 end

% build the adjacency matrix
 CKSym = cell(D,1);
 CAbs  = cell(D,1);
 LapN  = cell(D,1);
 LapUN = cell(D,1);
 
 % construct the normalized Laplacian 
 for ii=1:D
  [CKSym{ii},CAbs{ii}] = BuildAdjacency(thrC(SpC{ii},rho),K);
  DN = diag( 1./sqrt(sum(CKSym{ii})+eps) );
  LapN{ii} = speye(N) - DN * CKSym{ii} * DN;
  LapUN{ii} = sparse(diag(sum(CKSym{ii})) - CKSym{ii});
 end

