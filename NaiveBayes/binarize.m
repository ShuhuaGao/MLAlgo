function Xb = binarize(X)
%BINARIZE Transform the data into binary values (0 and 1).
%   criterion: if x > 0, then x is binarized to be 1; otherwise 0.
%USAGE: 
%   Xb = binarize(X), where X is the original data and Xb is the binarized data.
temp = X > 0;
Xb = double(temp);

end

