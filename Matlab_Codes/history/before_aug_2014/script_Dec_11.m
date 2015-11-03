clear all
close all
clc;
load('result_error.mat')
load('result_error_c.mat')
error_box = error';
error_box(:,[3:4]) = error_c';
figure(1);
boxplot(error_box, 'labels', {'view 1-clean', 'view 2-clean', 'view 1-corrupt', 'view 2-corrupt'})
grid on;
ylabel 'error rate'
%title('comparison of clean and dirty samples')