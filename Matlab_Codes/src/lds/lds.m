function [Yu, err] = lds(Xl,Xu,Yl,rho,opt)
% Yu = LDS(Xl,Xu,Yl,rho,opt)
%   Run the Low Density Separation algorithm as described in
%   "Semi-supervised classification by Low Density Separation" by
%   O. Chapelle and A. Zien
%
% Xl:  d x n matrix with the labeled points
% Xu:  d x m matrix with the unlabeled points
% Yl:  column vector of length n containing the labels (+1 or -1 for binary)
% rho: the constant in (2)
% opt: optional structure containing the (optional) fields,
%         C:       the soft margin parameter relative to 1/var^2
%                  [default = 1]
%         nofNn:   number of NN in the graph construction 
%                  [default = 0, i.e. fully connected graph]
%         sigma:   the width of the RBF kernel appearing at the end
%                  of 2.1.2, relative to the default value [default = Inf] 
%         delta:   the threshold in (6) [default = 0.1]
%         nofIter: number of iterations for C* to reach C [default = 10]
%         Cinit:   initial value for C* relative to C [default = 0.01] 
%         Cfinal:  final value of C* relative to C [default = 1]
%         opt_tb:  uses the optimization toolbox if available [default = 1]
%         maxiter: maximum number of iterations in each gradient
%                  descent (multiplied by nb of variables) [default = 3]
%         tolfun:  stopping criterion on the function value
%                  (relative to C) [default = 1e-5]
%         verb:    verbosity [default = 1]
%         splits:  estimate the test error by cross-validation
%                  splits is a cell array containing the indices of the points
%                  left out [default = leave-one-out].
% Yu: is a real valued output
% err: estimated error by cross-validation [cf opt.splits]
%  
% (C) O. Chapelle and A. Zien, MPI for biol. Cybernetics, Germany


% Fill up default values for opt
if ~exist('opt','var')
  opt = [];
end;
if ~isfield(opt,'delta')       opt.delta       = 0.1;       end;
if ~isfield(opt,'nofIter')     opt.nofIter     = 10;        end;
if ~isfield(opt,'C')           opt.C           = 1;         end;
if ~isfield(opt,'Cinit')       opt.Cinit       = 0.01;      end;
if ~isfield(opt,'Cfinal')      opt.Cfinal      = 1;         end;
if ~isfield(opt,'nofNn')       opt.nofNn       = 0;         end;
if ~isfield(opt,'sigma')       opt.sigma       = Inf;       end;
if ~isfield(opt,'opt_tb')      opt.opt_tb      = 1;         end;
if ~isfield(opt,'maxiter')     opt.maxiter     = 3;         end;
if ~isfield(opt,'tolfun')      opt.tolfun      = 1e-5;      end;
if ~isfield(opt,'verb')        opt.verb        = 1;         end;

% Check the arguments
[ d0, m0 ] = size( Xl );
[ d1, m1 ] = size( Xu );
if (d0 ~= d1) | ((m0 ~= length(Yl)) & (m0 + m1 ~= length(Yl)))
  error('Dimensions mismatch');
end;
m = m0 + m1;
X = [ Xl , Xu ];
clear Xl;
clear Xu;

% assert( 0 <= opt.delta && opt.delta < 1 );
assert( 1 <= opt.nofIter );
assert( 0 < opt.C );
assert( 0 < opt.Cinit && opt.Cinit <= opt.Cfinal );
assert( 0 <= opt.nofNn && opt.nofNn <= m );
assert( 0 < opt.sigma );

if (size(Yl,2) ~= 1)
  error(['Third argument should be the label vector']);
end;
ks = unique( Yl );
k = length( ks );
if( k < 2 )
  error(['At least two different classes (labels in Yl) required']);
end;
if( k == 2 )
  if any(abs(Yl)~=1)
    error(['Labels in binary problems must be +/-1']);
  end;
end;

if opt.verb >= 2
  fprintf('%d classes, %d labeled points, %d unlabeled, in %d dimension(s)\n',k,m0,m1,d0);
end;

% Compute the full distance graph ...
param.annEps = -1;  % do not use ANN (approximate NN)
if opt.verb >= 1
  if opt.nofNn == 0   
    fprintf('Computing  distances ...\n');
  else
    fprintf('Computing %d nearest neighbors graph ...\n', opt.nofNn);
    end;
end;
[ NN, D2 ] = calcNnDists( X, [], opt.nofNn, param.annEps );

% ... and the new metric
param.pathNorm = rho;
param.sigma = +inf;
param.nofNn = opt.nofNn;
if opt.verb >= 1
  fprintf('Computing shortest paths ...\n');
end;
E2 = graphDistKernelC( X, D2, NN, 1:m, param );
if( opt.nofNn ~= 0 )
  E2 = min( E2,  E2' );
  E2( isinf( E2 ) ) = 2 * max( E2( ~isinf( E2 ) ) );
end;

% Compute the new kernel, do the MDS reduction, ...
defaultSigma = calcDefaultSigma( E2(:), 2 );
if opt.verb >= 2
  fprintf( '  default sigma = %.3f\n', defaultSigma );
end;
sigma = opt.sigma * defaultSigma;
if isinf(sigma)
  if any(isinf(E2(:)))
    error(['Cannot do the MDS: the graph is not connected'])
  end;
  [Xmds,E] = mds(E2,opt);
else
  param.rbf = 'gauss';
  K = calcRbfKernel( E2, param.rbf, sigma, 1 );
  [Xmds,E] = mds(2-2*K,opt);
end;

% ... and find the cut p as in (6)
if opt.delta >= 0
  cond1 = 1-cumsum(E)/sum(E) < opt.delta;
  cond2 = E < opt.delta*E(1);
  nb_comp = min( find( cond1 & cond2 ) ) - 1;
else
  nb_comp = length(E);
end;
if opt.verb >= 2
  fprintf('  %d components kept\n',nb_comp);
end;

Xnldr = Xmds(:,1:nb_comp); % Keep only the first components
clear Xmds;
        
% Compute the different values of C
defaultC = 1 / sum(var(Xnldr)); % Default value of C = invert of the variance
if opt.verb >= 2
  fprintf( '  default C = %.3f\n', defaultC );
end;
C = opt.C * defaultC;
opt.Cinit = C * opt.Cinit;
opt.Cfinal = C * opt.Cfinal;
opt.C = C;

opt.s = 3; % The constant in L* (4)
          
% Train the TSVM
if opt.verb >= 1
  fprintf('Training TSVM')
end;
if opt.verb >= 2
  fprintf('\n')
end;

Yu = train_one_split( Xnldr, Yl, 1:m0, opt);

if nargout==2  % Do the cross-validation
  if ~isfield(opt,'splits') % Default = leave-one-out
    for i=1:m0
      opt.splits{i} = i;
    end;
  end;
  for i = 1:length(opt.splits)
    if opt.verb >= 1
      fprintf( 'Training split %d\n', i );
    end;
    val = unique(opt.splits{i});
    train = 1:m0;
    train(val) = [];
    [Yu2, obj(i,:)] = train_one_split( Xnldr, Yl(train), train, opt);
    if ( k == 2)
      err(i) = mean( Yl( val ) .* Yu2( 1:length(val) ) < 0);
    else
      [foo, ind] = max( Yu2, [], 2);
      ks2 = unique( Yl( train ) );
      err(i) = mean( Yl( val ) ~= ks2( ind( 1:length(val) ) ) );
    end;
  end;
  % Add a very tiny bit of the objective function in order to break the
  % possible ties during the model selection.
  err = err' + 1e-10*mean(obj(:)) / defaultC;
end;

function [Yu, obj] = train_one_split( Xnldr, Yl, lab, opt)
  ks = unique( Yl );
  k = length( ks );
  Y0 = zeros( size( Xnldr,1), 1 );
  if( k == 2 )
    if length( Yl ) > length ( lab )
      % For debugging: checking local minimum problems
      warning(['Initializing with the true labels of the unlabeled points']);
      opt.Cinit = opt.Cfinal; opt.nofIter = 1;
      [ w0, b, Yu, obj ] = primal_tsvm( Xnldr, Yl, opt); 

      Y0( lab ) = Yl ( lab );
      [ w, b, Yu, obj ] = primal_tsvm( Xnldr, Y0, opt, [w0; b]);
      fprintf('   Angle between both normal vectors = %f\n',w0'*w/norm(w0)/norm(w));
    else
      Y0( lab ) = Yl;
      [ w, b, Yu, obj ] = primal_tsvm( Xnldr, Y0, opt);
    end;
  else
    for( c = 1:k )
      if opt.verb >= 1
        fprintf( 'Training class #%d (label "%d")\n', c, ks(c) );
      end;
      Y0( lab ) = 2*( Yl == ks(c) ) - 1;
      [ w, b, Yu(:,c), obj(c) ] = primal_tsvm( Xnldr, Y0, opt);
    end;
    obj = sum( obj );
 end;


function [ w, b, Yu, obj ] = primal_tsvm( X, Y, opt, w0);
% === Solve the TSVM problem in the primal

  X = [X ones(length(Y),1)]; % Append one dimension for b

  % Global variables (not avoid giving them as arguments to obj_fun)
  global X_;  X_ =X;         
  global Xu_; Xu_=X(Y==0,:)'; % To speed-up obj_fun
  global R; % Rotation matrix to enforce constraint (5)
  
  % At each step, C* will be multiplied by exponent (2 in the paper)
  exponent = (opt.Cfinal/opt.Cinit)^(1/(opt.nofIter+1));
  C = opt.C;
  C2 = opt.Cinit;

  n = size(X,2);
  
  % We want to enforce vbal'*w = cbal
  vbal = ones(1,size(X,1));
  if any(Y==0)
    vbal(Y~=0) = 0;
  end;
  vbal = vbal*X;
  cbal = mean(Y(Y~=0)) * sum(vbal);
  % That can be done by rotating w in a new basis which has as
  % first component vbal. The first component of w in the new basis
  % in then fixed to w1
  [R,foo1,foo2] = svd(vbal'); clear foo1 foo2;
  w1 = cbal/(vbal*R(:,1));
  
  if nargin<4    % If w0 is given, start with it.
    w = zeros(n-1,1);
  else
    w = R(:,2:end)'*w0;
  end;
  
  % Check if the optimization toolbox if available
  opt_tb = opt.opt_tb & license('checkout','Optimization_toolbox');
  
  % Initialize the options for the optimizer
  maxiter = opt.maxiter*size(X,1);       % Maxinum number of iterations
  tolfun = opt.tolfun*C;                 % Stopping criterion
  if opt_tb
    opt2 = optimset('GradObj','on','LargeScale','off',...
                    'Display','off','TolFun',tolfun,'MaxIter',maxiter);
    if opt.verb >= 3
      opt2 = optimset(opt2,'Display','iter');
    end;
  else
    opt2.length = maxiter;
    opt2.tolX = 0; 
    opt2.tolFun = tolfun;
    opt2.verb = opt.verb - 1;
  end; 
  
  % Main loop
  for i = 0:opt.nofIter
    C2 = C2*exponent;
    if opt.verb == 1, fprintf('.'); end;
    
    if opt_tb
      [w,obj,flag,out,grad] = fminunc(@obj_fun,w,opt2,w1,Y,C,C2,opt.s);
      iter = out.iterations;
    else
      [w,fX,count] = minimize(w,@obj_fun,opt2,w1,Y,C,C2,opt.s);
      iter = length(fX);
      [obj, grad] = obj_fun(w,w1,Y,C,C2,opt.s);
      flag = (iter < opt2.length);
    end;

    if opt.verb >= 2
      fprintf(['  Iter = %d\t C* = %f\t Obj = %f [%d iterations, ' ...
               'norm of grad = %f]\n'],i,C2,obj,iter,norm(grad));
    end;
 
    if ~flag & (opt.verb >= 1)
       warning('Maximum number of iterations reached');
    end;
  end;
  
  % Rotate back w and get w and b
  w = R*[w1; w];  
  Yu = Xu_'*w;
  b = w( n );
  w = w( 1:n-1 );

   if opt.verb == 1, fprintf('\n'); end;
   if opt.verb >= 1, fprintf('Done !\n'); end;

function [obj,grad] = obj_fun(w,w1,Y,C,C2,s)
% Cost function (4)
  global X_ Xu_ R
  
  w = R*[w1; w];
  
  out = X_*w;
  lab   = find(Y~=0);
  unlab = find(Y==0);
  sv = find((out.*Y < 1) & (Y~=0));
  
  cost_unlab = exp(-s*out(unlab).^2);
  w0 = w; 
  w0(end)=0;
  
  obj = .5*w0'*w0;
  obj = obj + C *sum((1-out(sv).*Y(sv)).^2);
  obj = obj + C2*sum(cost_unlab);
  
  grad = w0;
  grad = grad + 2*C *X_(sv,:)'*(out(sv)-Y(sv));
  tmp = 2*C2*s*cost_unlab.*out(unlab);
  grad = grad - Xu_*tmp;

  grad = grad'*R; grad = grad(2:end)';



function [Y, L] = mds(D,opt)
% Classical Multidimensional Scaling.

if opt.verb >= 1, fprintf('Computing MDS ...\n'); end;
n = length(D);

if opt.delta>=0
  H = eye(n) - repmat(1/n,n,n);
  K = H * (-.5 * D) * H;
  [V L] = eig((K+K')./2);
  [L ind] = sort(-diag(L)); 
  L = -L;
  
  keep = find(L > 1e-10*max(L)); 
  Y = V(:,ind(keep)) * diag(sqrt(L(keep)));
else
  k = -opt.delta;
  H = eye(n) - repmat(1/n,n,n);
  K = H * (-.5 * D) * H;
  K = (K+K')/2;
  
  [V L] = eig(K(1:k,1:k));
  L = diag(L);
  keep = find(L > 1e-10*max(L)); 
  V = V(:,keep) .* repmat(L(keep)'.^(-1/2),k,1);
  Y = K(:,1:k)*V;
  L = L(keep);
end;
  
  

