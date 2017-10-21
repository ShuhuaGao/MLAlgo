% the MINIST data is originally 784-dimensional
% apply PCA to reduce the dimension to 2d and 3d
% then visualize the reduced data

close all; clear;

%% load data: trainX, trainy, testX, testy (each column is a data sample)
load('../Data/MNIST/MNIST.mat')

%% apply PCA to trainX
% 2d
[P, ~] = PCA(trainX, 2);
trainX2d = P' * trainX;
figure;
gscatter(trainX2d(1,:), trainX2d(2,:), trainy);
xlabel('PC1');
ylabel('PC2');


