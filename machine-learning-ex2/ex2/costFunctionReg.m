function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
l = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = transpose(sigmoid(transpose(theta)*transpose(X)));
J = (dot(-y, log(h))-dot((ones(m,1)-y), log(ones(m,1)-h)))/m + lambda/(2*m)*dot(ones(l-1,1), (theta(2:l)).^2);

temp = theta;
temp(1,1) = 0;
grad = (transpose(h-y)*X)/m + transpose((lambda/m) * temp);


% =============================================================

end
