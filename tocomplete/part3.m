%%% 1. Clear variables and close all figures
clearvars;
close all;

%%% 2. Define input training examples
X = [0 1; 1 1; 1 0; 0 0];
y = [1;0;1;0];

%%% 3. Train a non-linear SVM classifier
%%% PLACE YOUR CODE HERE
svm = fitcsvm(X, y, ...             % Train SVM on X and y
              'KernelFunction', 'rbf', ... % Use radial basis function (Gaussian) kernel
              'KernelScale', 'auto');      % Let MATLAB choose the kernel scale automatically


%%% 4. Compute decision regions
%%% PLACE YOUR CODE HERE
%%% Hint: Use plot_XOR_and_regions as inspiration
d = 0:0.01:1;                      % Define a grid from 0 to 1 in steps of 0.01
[X1, X2] = meshgrid(d, d);        % Create a full 2D grid of X1 and X2 values
XGrid = [X1(:), X2(:)];           % Reshape the grid into a list of coordinates (Nx2)
[~, scores] = predict(svm, XGrid);% Predict the class scores for the grid using the SVM
decision_values = scores(:, 2);   % Extract the decision boundary values for the positive class

figure; hold on;                                 % Create a figure and enable multiple plots
gscatter(X(:,1), X(:,2), y, 'rb', 'xo');          % Plot original XOR training points
contour(X1, X2, reshape(decision_values, size(X1)), [0,0], 'k'); % Plot decision boundary
title('Non-linear SVM XOR Decision Boundary');   % Add a title
legend('Class 1', 'Class 0', 'Decision Boundary');% Add legend

% Should work now