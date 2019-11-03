function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% Calculate h to be an m * 1 size vector/matrix
h = transpose(sigmoid(transpose(theta)*transpose(X)));
J = (dot(-y, log(h))-dot((ones(m,1)-y), log(ones(m,1)-h)))/m;

% Use vector operation to get gradient
grad = (transpose(h-y)*X)/m;











% =============================================================

end
