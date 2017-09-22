function nb = trainBernoulliNB(nb, Xtrain, ytrain)
%trainBernoulliNB Train a Bernoulli-Beta Naive Bayes Classifier
%INPUT:
%   nb: a Bernoulli-Beta Naive Bayes Classifier
%   Xtrain: input of training set, each row is an example (binarized)
%   ytrain: output of training set, each row is a class label
%OUTPUT:
%   nb: a trained Bernoulli-Beta Naive Bayes Classifier with the following fields
%   -'a': a
%   -'b': b
%   -'alpha': ML estimation of class prior 
%   -'mu': MAP estimation of conditionally independent feature Bernoulli distribution

% ML estimation of class prior
N = length(ytrain);
m = size(Xtrain, 2); % number of features
Ns = sum(ytrain);
nb.alpha = Ns / N;

% MAP estimation for the features
% 1. split the dataset according to the class label
% 2. estimate the Bernoulli parameter for each feature in each label
nb.mu = NaN(m, 2);
for c = [0, 1] % class 0 and 1
    Dc = Xtrain(ytrain == c, :);
    Nc = size(Dc, 1);
    for j = 1:m % each feature
        ajc = nb.a(j, c + 1);
        bjc = nb.b(j, c + 1);
        Njc1 = sum(Dc(:,j) == 1);
        nb.mu(j,c + 1) = (Njc1 + ajc - 1) / (Nc + ajc + bjc - 2);
    end
end
    
end

