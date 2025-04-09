%%% 1. Initial cleanup, add paths and load in data
%%% DON'T CHANGE
clearvars;
close all;
addpath('../data');
addpath('../helper');
load lab3cardata.mat;

%%% 2. Declare total number of input neurons, hidden layer neurons and output
%%% neurons

%%% DON'T CHANGE
input_neurons = 6;

%%% This we can change
hidden_neurons = 4;

%%% DON'T CHANGE
output_neurons = 4;

%%% 3. Compute the total weights between the input and hidden layer and
%%% the hidden layer and output layer.  Also compute the total amount
%%% of weights
%%% DON'T CHANGE
total_weights_W1 = (input_neurons + 1)*hidden_neurons;
total_weights_W2 = (hidden_neurons + 1)*output_neurons;
total_weights = total_weights_W1 + total_weights_W2;

%%% 4. Create the initial parameter vector of weights
%%% DON'T CHANGE
rng(123);
e_init_1 = sqrt(6) / sqrt(input_neurons + hidden_neurons);
e_init_2 = sqrt(6) / sqrt(hidden_neurons + output_neurons);
initial_vec = zeros(total_weights,1);
initial_vec(1:total_weights_W1) = 2*e_init_1*rand(total_weights_W1,1) - e_init_1;
initial_vec(total_weights_W1 + 1:end) = 2*e_init_2*rand(total_weights_W2,1) - e_init_2;

%%% 5. Set total number of iterations
%%% DON'T CHANGE
N = 400;

%%% 6. Regularization parameter - This you can change
lambda = 0;

%%% 7. Declare optimization settings
%%% PLACE YOUR CODE HERE

%%% 8. Find optimal weights
%%% PLACE YOUR CODE HERE
%%% MAKE SURE THE OUTPUT WEIGHT PARAMETER VECTOR IS STORED IN A VARIABLE CALLED weights

%%% 9. Extract out the final weight matrices
%%% DON'T CHANGE
W1 = reshape(weights(1:total_weights_W1), hidden_neurons, input_neurons + 1).';
W2 = reshape(weights(total_weights_W1+1:end), output_neurons, hidden_neurons + 1).';

%%% 10. Compute predictions for training and testing data
%%% PLACE YOUR CODE HERE

%%% 11. Compute classification accuracy for training and testing data
%%% PLACE YOUR CODE HERE