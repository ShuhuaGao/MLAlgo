function lrc = logisticRegressionClassifier()
%LOGISTICREGRESSIONCLASSIFIER Create a classifier based on logistic regression
%USAGE:
%lrc = logisticRegressionClassifier() returns a struct with two fields, w and lambda. w is the
%parameter vector which is learned during training and lambda is the l2 regularization weight
%which is specified during training.

lrc = struct('w', [], 'lambda', []);

end

