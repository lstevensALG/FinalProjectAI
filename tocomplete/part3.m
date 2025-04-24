%%% 1. Clear variables and close all figures
clearvars;
close all;

%%% 2. Define input training examples
X = [0 1; 1 1; 1 0; 0 0];
y = [1;0;1;0];

%%% 3. Train a non-linear SVM classifier
%%% PLACE YOUR CODE HERE
svm_classifier = fitcsvm(X, y, 'KernelFunction', 'rbf', 'KernelScale', 'auto');

%%% 4. Compute decision regions
%%% PLACE YOUR CODE HERE
%%% Hint: Use plot_XOR_and_regions as inspiration
decision = 0:0.01:1;
[X1, X2] = meshgrid(decision, decision);
XGrid = [X1(:), X2(:)];
[~, scores] = predict(svm_classifier, XGrid);
decision_values = scores(:, 2);

figure; hold on;
gscatter(X(:,1), X(:,2), y, 'rb', 'xo');
contour(X1, X2, reshape(decision_values, size(X1)), [0,0], 'k');
title('Non-linear SVM XOR Decision Boundary');
legend('Class 1', 'Class 0', 'Decision Boundary');