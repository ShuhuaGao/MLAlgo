% Apply PCA to reduce the dimensionality of raw data from 784 to 40, 80 and 200 respectively. 
% Classifying the test images using the rule of nearest neighbor. 

close all; clear;

%% load data: trainX, trainy, testX, testy (each column is a data sample)
load('../Data/MNIST/MNIST.mat')


%% apply PCA
accuracy = containers.Map('KeyType', 'double', 'ValueType', 'double');
for d = [40, 80, 200]
    [~, ~, trainXd] = PCA(trainX, d);
    [~, ~, testXd] = PCA(testX, d);
    testp = nan(size(testy));   % predication for test set
    % for each sample in testXd, find its nearest neighbor in trainXd
    for jj = 1: size(testXd, 2)
        diff = trainXd - testXd(:, jj);
        % we need to compute the 2-norm of each column of diff as the distance
        d2 = sum(diff.^2); % squared distance
        [~, minIndex] = min(d2);
        testp(jj) = trainy(minIndex);
    end
    % compute the accuracy
    accuracy(d) = sum(testp == testy) / length(testy);
end
% report 
fprintf('Accuracy is %f, %f and %f.\n', accuracy(40), accuracy(80), accuracy(200));
