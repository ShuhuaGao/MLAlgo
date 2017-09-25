function pred = predictLogisticRegressionClassifier(lrc, X)
%PREDICTLOGISTICREGRESSIONCLASSIFIER Use the given classifier to make predications
%   lrc: a given classifier
%   X: data matrix, whose row is a feature vector
%   pred: the predications, a column vector

% prepend 1 for each sample to allow the bias
X = [ones(size(X,1), 1), X];
p = standardLogistic(X * lrc.w);
pred = p > 0.5;

end

