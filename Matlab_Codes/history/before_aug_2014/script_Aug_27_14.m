% test on covariate shift

mu1 = [1.5;1.5];

mu2 = [-1.5;-1.5];


Sigma = [1, -0.5;-0.5, 1];

py = [0.5,0.5];
r1 = mvnrnd(mu1, Sigma, 1000*py(1));
r2 = mvnrnd(mu2, Sigma, 1000*py(2));

figure(1)
ax(1)= subplot(1,2,1);
plot(r1(:,1), r1(:,2),'+b');
hold on
plot(r2(:,1), r2(:,2),'*r');
hold off
xLimits = get(gca,'XLim');  %# Get the range of the x axis
yLimits = get(gca,'YLim');  %# Get the range of the y axis


py = [0.1,0.9];
r3 = mvnrnd(mu1, Sigma, 1000*py(1));
r4 = mvnrnd(mu2, Sigma, 1000*py(2));


ax(2)=subplot(1,2,2);
plot(r3(:,1), r3(:,2),'+b');
hold on
plot(r4(:,1), r4(:,2),'*r');
hold off
axis([xLimits, yLimits])
%linkaxes([ax(2) ax(1)],'xy');

%% logistic regression
py = [0.5,0.5];
r1 = mvnrnd(mu1, Sigma, 1000*py(1));
r2 = mvnrnd(mu2, Sigma, 1000*py(2));
r11 = [r1;r2];

w= [1;1];

f = @(y, x)(1./(1 + exp(-y.*(x*w))));

y = sign(f(ones(1000,1), r11)-0.5);

figure(2)
ax(3)= subplot(1,2,1);
plot(r11(find(y==1),1), r11(find(y==1),2),'+b');
hold on
plot(r11(find(y==-1),1), r11(find(y==-1),2),'*r');
hold off
xLimits = get(gca,'XLim');  %# Get the range of the x axis
yLimits = get(gca,'YLim');  %# Get the range of the y axis

theta = pi/3;
Q = [cos(theta) sin(theta); -sin(theta) cos(theta)];


r3 = mvnrnd(mu1, Q'*Sigma*Q, 1000*py(1));
r4 = mvnrnd(mu2, Q'*Sigma*Q, 1000*py(2));
r22 = [r3;r4];

y2 = sign(f(ones(1000,1), r22)-0.5);

ax(4)=subplot(1,2,2);
plot(r22(find(y2==1),1), r22(find(y2==1),2),'+b');
hold on
plot(r22(find(y2==-1),1), r22(find(y2==-1),2),'*r');
hold off
axis([xLimits, yLimits])


% linkaxes([ax(4) ax(3)],'xy');

