% (C) A. Zien and O. Chapelle, MPI for biol. Cybernetics, Germany

function [ K ] = calcRbfKernel( D2, rbfName, sigma, symmetrize );

% === params
if( ~ exist( 'symmetrize', 'var' ) )
  symmetrize = 0;
end;

% === compute kernel matrix (K)
switch( rbfName )
 case 'gauss'
  if( sigma == +inf )
    K = -D2;
    % if D2 is symetric, it should be 
    % K = -H*D2*H, where H = eye(n) - repmat(1/n,n,n);
  else
    assert( 0 < sigma && sigma < +inf );
    K = exp( (-0.5/sigma^2) * D2 );
  end;
 case 'laplace'
  assert( 0 < sigma && sigma < +inf );
  K = exp( (-1/sigma) * sqrt(D2) );
 otherwise
  error( 'unknown RBF' );
end;

% === symmetrize kernel
if( symmetrize )
  [ m1, m2 ] = size( K );
  if( m1 > m2 )
    K(1:m2,:) = ( K(1:m2,:) + K(1:m2,:)' ) ./ 2;
  else
    assert( m1 == m2 );
    K = ( K + K' ) ./ 2;
  end;
end;

% === check kernel matrix
% --- check if entries are finite
nofInfs = sum( isinf( K(:) ) );
nofNans = sum( isnan( K(:) ) );
if( nofInfs > 0 | nofNans > 0 )
  warning( sprintf( 'kernel improper: %d INFs, %s NANs', nofInfs, nofNans ) );
end;
% --- check for numerical deviations from symmetry
[ m1, m2 ] = size( K );
nofDiffs = sum( sum( K(1:m2,:) ~= K(1:m2,:)' ) );
if( nofDiffs > 0 )
  assert( ~ symmetrize );
  normDiff = norm( K(1:m2,:) - K(1:m2,:)' );
  if( normDiff >= 1e-12 )
    warning( sprintf( 'asymmetric kernel: %d differences, norm=%e', nofDiffs, normDiff ) );
  end;
  K(1:m2,:) = ( K(1:m2,:) + K(1:m2,:)' ) ./ 2;
end;


