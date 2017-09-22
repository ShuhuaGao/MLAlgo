function BNB = BernoulliNB(a, b)
%BERNOULLINB Create a Naive Bayes Classifier for binary classification. The features should
%also be binary.
%   The class prior is estimated with maximum likelihood. The Bernoulli distribution of the
%   features are estimated by maximum a posteriori using a Beta prior.
%INPUT:
%   a, b are matrices both of size number_of_features-by-2, where a(j,c) and b(j,c) are the shape
%   parameters of the Beta prior for MAP inference of the jth feature in class c.
%OUTPUT:
%   a struct representing a Bernoulli-Beta Naive Bayes Classifier with the following fields
%   -'a': a
%   -'b': b
%   -'alpha': ML estimation of class prior (to be assigned by training)
%   -'mu': MAP estimation of conditionally independent feature Bernoulli distribution (to be assigned by training)

BNB = struct('a', a, 'b', b, 'alpha', [], 'mu', []);

end

