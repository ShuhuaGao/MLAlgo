function y = kNNClassify(Xtrain, ytrain, X, Ks, distance)
%KNNCLASSIFY Classification with k-nearest neighbors
%   Xtrain and ytrain are the training set, with each row of Xtrain as an input feature vector
%   X is an input matrix for predication
%   each element of vector Ks is the number of nearest neighbors to be considered
%   distance is the distance function, i.e., 'Euclidean' or 'Hamming'
%   y is the predicated class label for each row of X, then arranged into a matrix where each
%   column represents the result for each K in Ks

y = nan(size(X, 1), length(Ks));
ds = nan(size(Xtrain, 1), 1); % distance of a row vector x from each row of Xtrain
if strcmp(distance, 'Euclidean')
    distFunc = @Euclidean;
elseif strcmp(distance, 'Hamming')
    distFunc = @Hamming;
else
    error('Only Euclidean and Hamming distances are supported!');
end

for ii = 1:size(X, 1)
    % compute distance
    dX = Xtrain - X(ii, :);
    for jj = 1: size(dX, 1)
        ds(jj) = distFunc(dX(jj, :));
    end
    % find the k nearest neighbors
    [~, indices] = sort(ds);
    for jj = 1:length(Ks)
        yK = ytrain(indices(1:Ks(jj))); % labels for the k nearest neighbors
        y(ii, jj) = mode(yK); % majority 
    end
end

end

function d = Euclidean(x)
d = norm(x);
end

function d = Hamming(x)
d = sum(x ~= 0);
end


