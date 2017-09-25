function P = standardLogistic(T)
%LOGISTIC Standard logistic function, P = logistic(T)
% If T is a vector or matrix, then P is also a vector computed in an element-wise way.
f = @(t) 1 / (1 + exp(-t));
P = arrayfun(f, T);
end
