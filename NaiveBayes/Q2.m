clear; close all;

%% load the data
load('../Data/spamData.mat');

%% z-normalization and log transform
Xtrain_z = standardize(Xtrain);
Xtest_z = standardize(Xtest);
Xtrain_l = logTransform(Xtrain);
Xtest_l = logTransform(Xtest);

%% Gaussian Naive Bayes
trainError = NaN(1, 2);
testError = NaN(1, 2);
% z normalization
nb = gaussianNB();
nb = trainGaussianNB(nb, Xtrain_z, ytrain);
predictTrain = testGaussianNB(nb, Xtrain_z);
trainError(1) = sum(predictTrain ~= ytrain) / length(ytrain);
predictTest = testGaussianNB(nb, Xtest_z);
testError(1) = sum(predictTest ~= ytest) / length(ytest);
% log transform
nb = gaussianNB();
nb = trainGaussianNB(nb, Xtrain_l, ytrain);
predictTrain = testGaussianNB(nb, Xtrain_l);
trainError(2) = sum(predictTrain ~= ytrain) / length(ytrain);
predictTest = testGaussianNB(nb, Xtest_l);
testError(2) = sum(predictTest ~= ytest) / length(ytest);

%% summary
fprintf('z-normalization:\n \ttrain error = %.5f, test error = %.5f\n', trainError(1), testError(1));
fprintf('log transform:\n \ttrain error = %.5f, test error = %.5f\n', trainError(2), testError(2));

