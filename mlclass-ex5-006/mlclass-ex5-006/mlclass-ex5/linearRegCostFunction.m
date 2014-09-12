function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = (theta' * X')';

temp = h - y;
temp = temp .^ 2;
J1 = sum(temp) / (2*m); 
J2 = lambda * sum(theta(2:end) .^ 2) / (2*m);
J = J1 + J2;

temp = h - y;
grad1 = (X' * temp)/m;
grad2 = lambda * theta / m;
grad(2:end) = grad1(2:end) + grad2(2:end);
grad(1) = grad1(1);

% =========================================================================

grad = grad(:);

end
