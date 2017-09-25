function lrc = trainLogisticRegressionClassifier(lrc, X, y, lambda)
%TRAINLOGISTICREGRESSIONCLASSIFIER Train a classifier based on logistic regression
%USAGE:
%lrc = trainLogisticRegressionClassifier(lrc, X, y)
%INPUT: X is a matrix where each row represents a feature vector while y is the label for each
%sample (0 or 1); lrc is a classifier to be trained; lambda is the l2 regularization weight.
%OUTPUT: the trained classifier

% prepend 1 for each sample to allow the bias
X = [ones(size(X,1), 1), X];
% build the negative log-likelihood function and its gradient, Hessian, which only depend on
% the parameter vector w
f = @(w) NL2(X, y, w, lambda);
g = @(w) gradient(X, y, w, lambda);
h = @(w) hessian(X,  w, lambda);
% Newton's method to learn the parameters
rng(1234);
w0 = rand(size(X, 2), 1) - 0.5;
epsilon = 1e-3;
wstar = descentNewton(f, g, h, w0, epsilon);
lrc.lambda = lambda;
lrc.w = wstar;
end



function [f, Lambda] = helper(X, w, lambda)
%HELPER a utility function to compute the logistic function values and the Lambda matrix
t = X * w; % column vector, each element of which is w'*Xi
f = standardLogistic(t);
Lambda = diag(zeros(length(w), 1) + lambda);
Lambda(1, 1) = 0;
end

function v = NL2(X, y, w, lambda)
%NL Compute the negative log-likelihood with l2 regularization
%INPUT:   w is the parameter vector and lambda is the l2 regularization weight. 
%OUTPUT: v is the negative log-likelihood function value

[f, Lambda] = helper(X, w, lambda);
t1 = y'*log(f);
t2 = (1-y)'*log(1-f);
NL = -(t1 + t2); %log is the natural logarithm
v = NL + 1/2*w'*Lambda*w; %NL2
end

function g = gradient(X, y, w, lambda)
[f, Lambda] = helper(X, w, lambda);
g = X'*(f-y) + Lambda*w;
end

function h = hessian(X,  w, lambda)
[f, Lambda] = helper(X, w, lambda);
S = diag(f .* (1-f));
h = X'*S*X + Lambda;
end


