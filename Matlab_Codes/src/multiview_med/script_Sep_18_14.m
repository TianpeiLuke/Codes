%% test on two-moon dataset
clearvars
close all
addpath('./libsvm-3.18/matlab/')
addpath('../../../../../../MATLAB/cvx/');

% num_pos=1000; 
% num_neg=1000;
% %radii=0.75+0.15*randn(num_pos+num_neg,1);
% num = num_pos+num_neg; 
% y=zeros(num,1);
% 
% scale = 2;
% radii=scale*ones(num,1);%+0.22*randn(num_pos+num_neg,1);
% phi  =rand(num_pos+num_neg,1).*pi;
%    dim = 2;        
%            % the following generates two half circles in two dimensions in two-moon form
%            x=zeros(dim,num);
%            for i=1:num_pos
%              x(1,i)=radii(i)*cos(phi(i));
%              x(2,i)=radii(i)*sin(phi(i));
%              y(i,1)=1;
%            end
%            for i=num_pos+1:num_neg+num_pos
%              x(1,i)=scale+radii(i)*cos(phi(i));
%              x(2,i)=-radii(i)*sin(phi(i))+ 0.5*scale;
%              y(i,1)=2;
%            end
% figure           
% plot(x(1,1:num_pos), x(2,1:num_pos), 'xb');
% hold on;
% plot(x(1,num_pos+1:num_neg+num_pos), x(2,num_pos+1:num_neg+num_pos), 'xr');
% 
% 
% figure
% data=dbmoon(20); 
% plot(data(find(data(:,3)==1),1),data(find(data(:,3)==1),2), 'xb' );
% hold on
% plot(data(find(data(:,3)==-1),1),data(find(data(:,3)==-1),2), 'or' );
% hold off
% 
% 
% 

% data=dbmoon(100,-2,6,3); 
% plot(data(find(data(:,3)==1),1),data(find(data(:,3)==1),2), 'xb' );
% hold on
% plot(data(find(data(:,3)==-1),1),data(find(data(:,3)==-1),2), 'or' );
% hold off
N1 = 200;
N = 2*N1;
data= dbmoon(N1,-4,4.25,2.5); 
figure
plot(data(find(data(:,3)==1),1),data(find(data(:,3)==1),2), 'xb' );
hold on
plot(data(find(data(:,3)==-1),1),data(find(data(:,3)==-1),2), 'or' );
hold off

X_L = data(:,1:2);
y_L = data(:,3);

ind_perm = randperm(length(y_L));
X_L(ind_perm,:) = X_L;
y_L(ind_perm) = y_L;

ind_c1 = ind_perm(1:N1);
ind_c2 = ind_perm(N1+1:N);

optionsvm = '-c 1 -g 0.5';
model = svmtrain(y_L, X_L,optionsvm);
 [Q1, ~] = qr(randn(2));
x1plot = linspace(min(X_L(:,1))-0.5, max(X_L(:,1))+0.5, 200)';
x2plot = linspace(min(X_L(:,2))-0.5, max(X_L(:,2))+0.5, 200)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
this_X = [reshape(X1,numel(X1),1), reshape(X2,numel(X2),1)];
[pred_label] = svmpredict(rand(size(this_X,1),1), this_X, model, '-q'); 
vals = reshape(pred_label, size(X1));
% vals = zeros(size(X1));
% for i = 1:size(X1, 2)
%    this_X = [X1(:, i), X2(:, i)]*Q1;
%    [pred_label] = svmpredict(rand(size(this_X,1),1), this_X, model, '-q'); 
%    vals(:, i) = pred_label;
%    
%    X1_n(:,i) = this_X(:,1);
%    X2_n(:,i) = this_X(:,2);
% end


% Plot the SVM boundary
figure(1)
contourf(X1, X2, vals);
hold on
plot(data(find(data(:,3)==1),1),data(find(data(:,3)==1),2), 'xy' );
plot(data(find(data(:,3)==-1),1),data(find(data(:,3)==-1),2), 'or' );
%contour(X1, X2, vals, [0 0], 'Color', 'b');
hold off;


L = 10;
ind_L_c1 = randsample(ind_c1, L);
ind_L_c2 = randsample(ind_c2, L);
model2 = svmtrain(y_L([ind_L_c1 ind_L_c2]), X_L([ind_L_c1 ind_L_c2],:),optionsvm);
vals2 = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   [pred_label] = svmpredict(rand(size(this_X,1),1), this_X, model2, '-q'); 
   vals2(:, i) = pred_label;
end

figure(2)
contourf(X1, X2, vals2);
hold on
plot(data(find(data(:,3)==1),1),data(find(data(:,3)==1),2), 'xy' );
plot(data(find(data(:,3)==-1),1),data(find(data(:,3)==-1),2), 'or' );
plot(X_L(ind_L_c1,1), X_L(ind_L_c1,2), '+c', 'Linewidth', 2, 'Markersize', 8)
plot(X_L(ind_L_c2,1), X_L(ind_L_c2,2), 'ow', 'Linewidth', 2, 'Markersize', 8)
%contour(X1, X2, vals, [0 0], 'Color', 'b');
hold off;








