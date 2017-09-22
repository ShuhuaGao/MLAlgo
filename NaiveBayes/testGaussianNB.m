function y = testGaussianNB(nb, X)
%testBernoulliNB Classify test input X with a Gaussian Naive Bayes Classifier
%INPUT:
%   nb: a Gaussian Naive Bayes Classifier
%   X: an input matrix, each row being a feature vector
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

% 1. class prior
if c == 1
    pc = nb.alpha;
else
    pc = 1 - nb.alpha;
end
p = log(pc);

% 2. feature likelihood
for jj = 1:length(x)
    mujc = nb.mu(jj, c + 1); % mean of the class conditioned normal distribution
    sjc = nb.sigmas(jj, c + 1); % variance
    xj = x(jj);
    pd = 1 / sqrt(2*pi*sjc) * exp(-(xj-mujc)^2 / (2*sjc));
    p = p + log(pd);
end
end

