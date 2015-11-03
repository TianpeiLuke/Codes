% Example on how to use LDS on the Uspst dataset

load uspst;

ks = unique( y );
k = length( ks );
for i=1:10
  Xl = X(idxLabs(i,:),:)';  
  Xu = X(idxUnls(i,:),:)';
  Yl = y(idxLabs(i,:));
  Yu = y(idxUnls(i,:));
  
  opt.C = 100; % large C performs best
  opt.delta = -50;
  opt.verb = 2;
  rho = 4;     % (cf figure 5)
  Yp = lds(Xl,Xu,Yl,rho,opt);
  [ dummy, Yc ] = max( Yp' );
  te(i) = mean( Yu ~= ks(Yc) );
  fprintf('    -> Test error after %d split(s): %f\n\n',i,mean(te));
end;


