clear all; close all;
f= @(x)(sin (pi/3*x));
ntr = 50;
nte = 1000;
sn = 0.2;

xtr = 12*sort(rand(ntr,1));
ytr = f(xtr) + randn(ntr,1)*sn;
xte = linspace(0,12,nte)';

cov = {@covSEiso}; sf = 1; ell = 0.4;

hyp0.cov  = log([ell;sf]);

mean = {@meanSum,{@meanLinear,@meanConst}}; a = 0; b =0;       % m(x) = a*x+b
hyp0.mean = [a;b];

lik_list= 'likGauss';
inf_list = 'infVB';

sdscale = 0.5;                  % how many sd wide should the error bars become?
col = {'k',[.8,0,0],[0,.5,0],'b',[0,.75,.75],[.7,0,.5]};
ymu{1} = f(xte); ys2{1} = sn^2; nlZ(1) = -Inf;
i= 1;

lik = lik_list;
hyp0.lik  = log(sn);
inf = inf_list;


Ncg = 150;
hyp = minimize(hyp0,'gp', -Ncg, inf, mean, cov, lik, xtr, ytr); % opt hypers
[ymu{i+1}, ys2{i+1}] = gp(hyp, inf, mean, cov, lik, xtr, ytr, xte);

figure, hold on
i=1
plot(xte,ymu{i},'Color',col{i},'LineWidth',2)
i=2
plot(xte,ymu{i},'Color',col{i},'LineWidth',2)
i=2
ysd = sdscale*sqrt(ys2{i});
fill([xte;flipud(xte)],[ymu{i}+ysd;flipud(ymu{i}-ysd)],...
col{i},'EdgeColor',col{i},'FaceAlpha',0.1,'EdgeAlpha',0.3);
leg = {'function'};
leg{2} = sprintf('%s/%s',...
lik_list,inf_list);
legend(leg)
plot(xtr,ytr,'k+'), plot(xtr,ytr,'ko'), legend(leg)