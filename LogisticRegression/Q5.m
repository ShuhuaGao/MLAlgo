close all; clear;
addpath('../NaiveBayes');
load('../Data/spamData.mat');

%% preprocess data
data = cell(3, 2);
preprocessors = {@standardize, @logTransform, @binarize};
for ii = 1:3
    data{ii, 1} = preprocessors{ii}(Xtrain);
    data{ii, 2} = preprocessors{ii}(Xtest);
end

%% logistic regression
lambdaList = [1:9, 10:5:100];
testErrors = NaN(length(lambdaList), 3);
trainErrors = NaN(length(lambdaList), 3);
for ii = 1:3    % each preprocessed data
    train = data{ii, 1};
    test = data{ii, 2};
    lrc = logisticRegressionClassifier();
    for jj = 1:length(lambdaList)   %each lambda
        lambda = lambdaList(jj);
        lrc = trainLogisticRegressionClassifier(lrc, train, ytrain, lambda);
        ptrain = predictLogisticRegressionClassifier(lrc, train);
        trainErrors(jj, ii) = sum(ptrain ~= ytrain) / length(ytrain);
        ptest = predictLogisticRegressionClassifier(lrc, test);
        testErrors(jj, ii) = sum(ptest ~= ytest) / length(ytest);
    end
end

%% plot
titles = {'z-normalization', 'log-transform', 'binarization'};
for ii = 1:3
    figure;
    plot(lambdaList, trainErrors(:, ii), lambdaList, testErrors(:, ii));
    grid on;
    xlabel('\lambda');
    ylabel('Error rate');
    title(titles{ii});
end
% or on a single figure
figure;
hold on;
for ii = 1:3
    plot(lambdaList, trainErrors(:, ii), lambdaList, testErrors(:, ii));
end
grid on;
xlabel('\lambda');
ylabel('Error rate');
legend('z-train', 'z-test', 'log-train', 'log-test', 'bi-train', 'bi-test');

