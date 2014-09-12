function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

yzeroindices = find(y == 0);
yoneindices = find(y == 1);

scatter(X(yzeroindices, 1),X(yzeroindices,2));
hold on;
scatter(X(yoneindices, 1),X(yoneindices,2),'+');

% =========================================================================



hold off;

end
