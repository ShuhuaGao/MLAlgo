function nb = gaussianNB( )
%GAUSSIANNB Create a Gaussian naive Bayes classifier
%   Both the class prior and feature distribution are estimated with maximum likelihood.
%USAGE:
%   nb = gaussianNB( )
%OUTPUT:
%   nb: a struct with three fields, to be learned by training
%   - 'alpha': class prior
%   - 'mu': expectation of the conditional Gaussian distribution of features
%   - 'sigmas': variance of the conditional Gaussian distribution of features.

nb = struct('alpha', [], 'mu', [], 'sigmas', []);

end

