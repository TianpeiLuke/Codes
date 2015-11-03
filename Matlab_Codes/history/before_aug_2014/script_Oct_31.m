%% Exp Oct 31
%% Test on the Bhattcharyya distance
clear all
close all
clc
% p1= rand(2,1);
% p1 = p1/sum(p1);
% p2 = rand(2,1);
% p2 = p2/sum(p2);
f = @(x,y) -log(sum(sqrt(x.*y))); %Bhattcharyya distance
load('exp1_Oct_31.mat');

r1 = linspace(0,1,100)';
r2 = sort(min([max([(r1' + 1e-2*randn(1,100)); zeros(1,100)]); ones(1,100)]),'ascend')';
u = 0.5*ones(2,1);
b_dist1 = zeros(100,1);
b_dist2 = zeros(100,100);

for t=1:length(r1)
   b_dist1(t) =  f((1-r1(t))*p1+r1(t)*u, (1-r1(t))*p2+r1(t)*u);
end

for t=1:length(r1)
    for s=1:length(r2)
       b_dist2(t,s) =  f((1-r1(t))*p1+r1(t)*u, (1-r2(s))*p2+r2(s)*u);
    end
end


figure(1)
plot(r1, b_dist1,'b','Linewidth',3.5);
xlabel 'ratio of uniform distribution'
ylabel 'Bhattacharyya distance'
grid on

figure(2)
mesh(r1, r2, b_dist2')
xlabel 'outlier ratio in view 1'
ylabel 'outlier ratio in view 2'
title('Bhattacharyya distance btw two views under various outlier ratio ')

