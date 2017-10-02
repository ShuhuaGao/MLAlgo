close all; clear;
addpath('../NaiveBayes');
load('../Data/spamData.mat');

%% preprocess data
% each row corresponds to a processing method
% for each row, the first element is the training set, and the 2nd one is the test set
data = cell(3, 2);
preprocessors = {@standardize, @logTransform, @binarize};
for ii = 1:3
    data{ii, 1} = preprocessors{ii}(Xtrain);
    data{ii, 2} = preprocessors{ii}(Xtest);
end

%% k nearest neighbors
Ks = [1:10, 15:100];
testErrors = nan(3, length(Ks));
trainErrors = nan(3, length(Ks));
distance = {'Euclidean', 'Euclidean', 'Hamming'};
for ii = 1:3    % 3 preprocessing
        train = data{ii, 1};
        test = data{ii, 2};
        predTrain = kNNClassify(train, ytrain, train, Ks, distance{ii}); % apply to training set
        predTest = kNNClassify(train, ytrain, test, Ks, distance{ii}); % apply to test set
        for jj = 1: length(Ks)
            trainErrors(ii, jj) = sum(predTrain(:,jj) ~= ytrain) / length(ytrain);
            testErrors(ii, jj) = sum(predTest(:,jj) ~= ytest) / length(ytest);
        end
    
end

%% Visualization
figure;
hold on;
for ii = 1:3
    plot(Ks, trainErrors(ii, :), Ks, testErrors(ii, :));
end
grid on;
xlabel('K');
ylabel('Error rate');
legend('z-train', 'z-test', 'log-train', 'log-test', 'bi-train', 'bi-test');
title('K-nearest neighbor classification');
% plot separately
preprocess = {'z-norm', 'log', 'binarize'};
for ii = 1:3
    figure;
    plot(Ks, trainErrors(ii,:), Ks, testErrors(ii,:));
    grid on;
    xlabel('K');
    ylabel('Error');
    title(preprocess{ii});
    legend('train', 'test');
end