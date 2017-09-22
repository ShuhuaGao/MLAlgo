function Xl = logTransform( X )
%logTransform transform each feature using log(xij + 0.1) (assume natural log)
%INPUT
%   X: a input dataset of which one row represents one observation
%OUTPUT
%   Xl: transformed data

Xl = log(X + 0.1);
end

