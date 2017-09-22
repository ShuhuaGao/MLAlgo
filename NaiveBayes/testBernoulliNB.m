function y = testBernoulliNB(nb, X)
%testBernoulliNB Classify test input X with a Bernoulli-Beta Naive Bayes Classifier
%INPUT:
%   nb: a Bernoulli-Beta Naive Bayes Classifier
%   x: a binarized input matrix, each row being a feature vector
%OUTPUT: 
%   y: a column vector labeling the class of the input

N = size(X, 1);
y = NaN(N, 1);

for ii = 1:N
    x = X(ii, :);
    if objective(nb, x, 1) > objective(nb, x, 0)
        y(ii) = 1;
    else
        y(ii) = 0;
    end
end
end

function p = objective(nb, x, c)
% OBJECTIVE compute the object function, i.e., class prior * likelihood.
% c: 0 or 1 for class 0 or class 1

% 1. class prior
if c == 1
    pc = nb.alpha;
else
    pc = 1 - nb.alpha;
end
p = pc;
% 2. feature likelihood
for j = 1: length(x)
    ujc = nb.mu(j, c + 1);
    if x(j) == 1
        p = p * ujc;
    else
        p = p * (1 - ujc);
    end       
end
end

