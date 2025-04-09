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

%%% 4. Perform One-Vs-All prediction on the training and test dataset
% Determine which class each of the examples in the test datasets are
% Create score matrices for both the training and test datasets
%%% PLACE YOUR CODE HERE

%%% 5. Calculate the classification accuracy for the training and test datasets
%%% PLACE YOUR CODE HERE