function nb = trainGaussianNB(nb, Xtrain, ytrain )
%trainGaussianNB Train a Gaussian naive Bayes classifier
%INPUT:
%   nb: a Gaussian Naive Bayes Classifier
%   Xtrain: input of training set, each row is an example (binarized)
%   ytrain: output of training set, each row is a class label
%OUTPUT:
%   nb: a trained Gaussian Naive Bayes Classifier with the following fields
%   - 'alpha': class prior
%   - 'mu': expectation of the conditional Gaussian distribution of features
%   - 'sigmas': variance of the conditional Gaussian distribution of features.

% ML estimation of class prior
N = length(ytrain);
m = size(Xtrain, 2); % number of features
Ns = sum(ytrain);
nb.alpha = Ns / N;

% feature Gaussian distribution: MLE
nb.mu = NaN(m, 2);
nb.sigmas = NaN(m, 2);
for c = [0, 1]
    Dc = Xtrain(ytrain == c, :); % a subset
    nb.mu(:, c+1) = mean(Dc);
    nb.sigmas(:, c+1) = var(Dc, 1);
end

end

