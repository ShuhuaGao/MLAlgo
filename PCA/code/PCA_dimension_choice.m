% Denoted the reduced dimension as d with d ¡Ü 784. Investigate what value of d preserves
% over 95% of the total energy after dimensionality reduction. Apply PCA to reduce the data
% dimension to d and report classification results based on nearest neighbor. Can you devise
% other criteria for automatically determining the value of d?

clear; close all;
%% load data: trainX, trainy, testX, testy (each column is a data sample)
load('../Data/MNIST/MNIST.mat')
% center data
trainX = trainX - mean(trainX, 2);


%% sigular values of trainX (centered)
s = svd(trainX);
lambda = s .^2; % scaled eigenvalues of the covariance matrix
% s is in descending order, we increase d until over 95% of the total energy
sl = 0;
for d = 1: length(lambda)
    sl = sl + lambda(d);
    if sl / sum(lambda) > 0.95
        break;
    end
end
fprintf('d >= %d\n', d);