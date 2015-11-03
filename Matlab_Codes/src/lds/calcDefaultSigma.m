% (C) A. Zien and O. Chapelle, MPI for biol. Cybernetics, Germany

function [ sigma ] = calcDefaultSigma( e2, k );

e2 = sort( e2 );
nl = length( e2 );
sigma2 = e2( 1 + round( nl / k ) );
if( sigma2 == +inf )
  idx = find( e2 < inf );
  if( length(idx) == 0 )
    sigma2 = 1;
  else
    sigma2 = max( e2(idx) );
    if( sigma2 <= 0 )
      sigma2 = 1;
    end;
  end;
end;
if( sigma2 <= 0 )
  idx = find( e2 > 0 );
  if( length(idx) == 0 )
    sigma2 = 1;
  else
    sigma2 = min( e2(idx) );
    if( sigma2 == +inf )
      sigma2 = 1;
    end;
  end;
end;
assert( 0 < sigma2 & sigma2 < +inf );
sigma = sqrt( sigma2 );


