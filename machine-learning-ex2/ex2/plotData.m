function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

positive = find(y==1);
negative = find(y==0);

plot(X(positive, 1), X(positive, 2), 'b+', 'MarkerSize', 10);
plot(X(negative, 1), X(negative, 2), 'ro', 'MarkerSize', 10);


hold off;

end
