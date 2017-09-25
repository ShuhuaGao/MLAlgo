clear; close all;
%% load the data
load('../Data/spamData.mat');

%% Binarization
Xtrain = binarize(Xtrain);
allOne = all(Xtrain);
Xtrain(:, allOne) = []; % remove columns that are all 1
Xtest = Xtest > 0;
Xtest(:, allOne) = [];

%% Train a Beta-bernoulli Naive Bayes Classifier 
% various parameter values for Beta prior
aList = 0:0.5:200;
trainErrorRate = NaN(1, length(aList));
testErrorRate = NaN(1, length(aList));
% for each set of parameter, build a NB, train and predict
for ii = 1:length(aList)
    a = zeros(size(Xtrain,2), 2) + aList(ii);
    nb = BernoulliNB(a, a);
    nb = trainBernoulliNB(nb, Xtrain, ytrain);
    trainPredict = testBernoulliNB(nb, Xtrain);
    testPredict = testBernoulliNB(nb, Xtest);
    trainErrorRate(ii) = sum(trainPredict ~= ytrain) / length(ytrain);
    testErrorRate(ii) = sum(testPredict ~= ytest) / length(ytest);
end

%% visualization
figure;
plot(aList, trainErrorRate, aList, testErrorRate);
grid on;
legend('train', 'test');
title('Error rate of Beta-bernoulli Naive Bayes');
xlabel('alpha');
ylabel('error rate');

%% Beta distribution with shape parameters both 0
a = 100;
b = 100;
x = linspace(0, 1, 100);
p = betapdf(x, a, b);
figure;
plot(x, p);
grid on;
title(sprintf('Beta distribution, a=%.2f, b=%.2f', a, b));

%% how about MLE of feature distribution
muMLE = NaN(size(Xtrain, 2), 2);
for c = [0, 1]
    Dc = Xtrain(ytrain==c, :);
    muMLE(:, c+1) = sum(Dc) / size(Dc, 1);
end
a = zeros(size(Xtrain,2), 2) + 5; % any number, unused
nb = BernoulliNB(a, a);
nb = trainBernoulliNB(nb, Xtrain, ytrain);
nb.mu = muMLE;
trainPredict = testBernoulliNB(nb, Xtrain);
testPredict = testBernoulliNB(nb, Xtest);
trainErrorRateMLE = sum(trainPredict ~= ytrain) / length(ytrain);
testErrorRateMLE = sum(testPredict ~= ytest) / length(ytest);
fprintf('With MLE of the feature distribution, train error rate = %.4f, test error rate = %.4f\n',...
    trainErrorRateMLE, testErrorRateMLE);
