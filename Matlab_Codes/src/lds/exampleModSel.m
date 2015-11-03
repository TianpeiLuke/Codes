load g50c;

i = 1;
Xl = X(idxLabs(i,:),:)';  
Xu = X(idxUnls(i,:),:)';
Yl = y(idxLabs(i,:));
Yu = y(idxUnls(i,:));

opt.C = 100; % large C performs best
opt.delta = -50; % Take only 50 components to go faster
opt.splits = mat2cell(1:50,1,10*ones(1,5)); % 5 fold cross-validation

for i=1:6
  rho = 4^(i-2);
  [Yp,err] = lds(Xl,Xu,Yl,rho,opt);
  te(i) = mean( Yp.*Yu < 0 );
  cv(i) = mean(err);
end;

[te; cv] % Compare the test and cv errors for the diffirent values of rho
