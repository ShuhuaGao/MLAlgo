function Xs = standardize( X )
%STANDARDIZE standardize each column so they have 0 mean and unit variance
%INPUT
%   X: a input dataset of which one row represents one observation
%OUTPUT
%   Xs: standardized data

Xs = (X - mean(X)) ./ std(X, 1);


end

