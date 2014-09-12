function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = (theta' * X')';
h = sigmoid(z);
J1 = sum((-y .* log(h)) - ((1 - y) .* log(1-h)))/m;
J2 = lambda * sum(theta(2:end) .^ 2) / (2*m);
J = J1 + J2;

temp = h - y;
delta = zeros(m, size(X,2));
    
for i=1:m,
    delta(i, :) = temp(i) * X(i,:);
end;
    
grad = sum(delta)/m + lambda * theta' / m;
temp = sum(delta)/m;
grad(1) = temp(1);
grad = grad';



% =============================================================

end
