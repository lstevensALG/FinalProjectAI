%%% 1. Load in the data
close all;
clearvars;
load fisheriris;

%%% 2. Create the training example data and expected labels
% This is a 150 training example dataset with 4 features
% stored in meas - 150 x 4
% The labels are stored as strings in a cell array called species
% Get the first 100 examples - 2 classes
% Look at two features for now - The last two
X = meas(1:100,3:4);

% Make the first class positive and the second class negative
y = [ones(50,1); zeros(50,1)];

%%% 3. Show what the data looks like
figure;

% Plots a scatter plot where each training example is highlighted in a
% different colour depending on the class
plot(X(1:50,1), X(1:50,2), 'b.', X(51:100,1), X(51:100,2), 'r.',...
    'MarkerSize', 16);

%%% 4. Train a linear SVM classifier
svm = fitcsvm(X, y, 'ClassNames', [0 1]);

%%% 5. Plot the support vectors as black circles on top of the data
% Get the support vectors
% This is a matrix of points that tells you the training examples that were
% selected as support vectors
sv = svm.SupportVectors;
hold on;
plot(sv(:,1), sv(:,2), 'ko', 'MarkerSize', 10);
legend('Positive Class', 'Negative Class', 'Support Vectors');

%%% 6. Declare new examples for predicting, then use SVM model to predict 
%%% new instances
newX = [1.7 2; 4.5 1.2];
labels = predict(svm, newX);