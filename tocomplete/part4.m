%%% 1. Initial cleanup, add paths and load in data
%%% DON'T CHANGE
clearvars;
close all;
addpath('../data');
addpath('../helper');
load lab3cardata.mat;

%%% 2. Create helpful variables
mTrain = size(Xtrain, 1); % Total number of training examples
mTest = size(Xtest, 1); % Total number of test examples
n = 4; % Total number of classes

%%% 3. Train non-linear SVM classifiers - one vs all using the training data
% Create cell arrays to store each SVM classifier
% Use the Gaussian kernel function
%%% PLACE YOUR CODE HERE
svm_classifiers = cell(n,1);           % Create a cell array to hold 4 individual SVM classifiers
for i = 1:n
    Ybinary = (Ytrain == i);           % Convert labels into binary (1 for class i, 0 otherwise)
    svm_classifiers{i} = fitcsvm(Xtrain, Ybinary, ... % Train a binary SVM for class i vs all others
                      'KernelFunction', 'rbf', ... % Use Gaussian (RBF) kernel
                      'KernelScale', 'auto', ...   % Let MATLAB pick the kernel scale
                      'ClassNames', [false true]); % Define class labels explicitly
end

%%% 4. Perform One-Vs-All prediction on the training and test dataset
% Determine which class each of the examples in the test datasets are
% Create score matrices for both the training and test datasets
%%% PLACE YOUR CODE HERE
scores_train = zeros(mTrain, n);        % Preallocate score matrix for training
scores_test = zeros(mTest, n);          % Preallocate score matrix for testing
for i = 1:n
    [~, score_tr] = predict(svm_classifiers{i}, Xtrain); % Get scores for training set
    [~, score_te] = predict(svm_classifiers{i}, Xtest);  % Get scores for testing set
    scores_train(:,i) = score_tr(:,2);        % Keep only the positive class score
    scores_test(:,i) = score_te(:,2);         % Do the same for test set
end

%%% 5. Calculate the classification accuracy for the training and test datasets
%%% PLACE YOUR CODE HERE
pred_train = predict_class(scores_train); % Choose the class with the highest score for each training example
pred_test = predict_class(scores_test);   % Do the same for testing set


accuracy_train = mean(pred_train == Ytrain) * 100; % Percent correct predictions for training set
fprintf('Multi-class SVM Training Accuracy: %.2f%%\n', accuracy_train); % Display training accuracy

accuracy_test = mean(pred_test == Ytest) * 100;    % Percent correct predictions for testing set
fprintf('Multi-class SVM Testing Accuracy: %.2f%%\n', accuracy_test);   % Display testing accuracy