% Example on how to use LDS on the g50c dataset

load g50c;

for i=1:10
  Xl = X(idxLabs(i,:),:)';  
  Xu = X(idxUnls(i,:),:)';
  Yl = y(idxLabs(i,:));
  Yu = y(idxUnls(i,:));
  
  opt.C = 0.1; % For this dataset, the classes overlap and it's
               % better to have a small value of C
  rho = 1;     % Anyvalue smaller than 3 is good (cf figure 5)
  Yp = lds(Xl,Xu,Yl,rho,opt);
  
  te(i) = mean( Yp.*Yu < 0 );
  fprintf('    -> Test error after %d split(s): %f\n\n',i,mean(te));
end;

