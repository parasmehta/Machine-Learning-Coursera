function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


    h = (theta' * X')';
    temp = h - y;
    mydelta = zeros(m, size(X,2));
    
    for i=1:m,
        mydelta(i, :) = temp(i) * X(i,:);
    end;
    
    deltasum = sum(mydelta);
    deltasum = deltasum * alpha;
    deltasum = deltasum/m;
    deltasum = deltasum';
    theta = theta - deltasum;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
