function [P, D] = PCA(X, m)
%PCA Principle component analysis
%INPUT
%   X: data matrix, where each column is a data sample
%OUTPUT
%   P: a matrix of m columns, representing the first m principle directions
%   D: a vector containing the m eigenvalues corresponding to the first m principle directions

% 1. center the data
X = X - mean(X);
% 2. apply SVD
[U, S, ~] = svd(X, 'econ');
% 3. the first m princple axes
P = U(:, 1:m);
D = diag(S);
D = D(1:m) .^ 2;

end

