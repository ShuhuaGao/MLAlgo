function [P, D, Xr] = PCA(X, m)
%PCA Principle component analysis
%INPUT ARGUMENTS
%   X: data matrix, where each column is a data sample
%OUTPUT ARGUMENTS
%   P: a matrix of m columns, representing the first m principle directions
%   D: a vector containing the m eigenvalues corresponding to the first m principle directions
%   Xr: data with reduced dimension, Xr = P' * X
%   

% 1. center the data
X = X - mean(X, 2);
% 2. apply SVD
[U, S] = svd(X, 'econ');
% 3. the first m princple axes
P = U(:, 1:m);
D = diag(S);
D = D(1:m) .^ 2;
% reduce
Xr = P' * X;
end

