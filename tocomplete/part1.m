%%% 1. Clear all variables and close all figures
%%% DON'T CHANGE
clearvars;
close all;
addpath('../helper');

%%% 2. Input training examples
%%% DON'T CHANGE
X = [0 1; 1 1; 1 0; 0 0];
y = [1;0;1;0];

%%% 3. Initialize weight matrices

%%% Number of input neurons
%%% DON'T CHANGE
input_neurons = 2;

%%% Number of hidden layer neurons
%%% This you can change
hidden_neurons = 2;

%%% Number of output layer neurons
%%% DON'T CHANGE
output_neurons = 1;

%%% DON'T CHANGE
% W1 is a 3 x X matrix - 2 + 1 input neurons, X hidden layer neurons
rng(123);
e_init_1 = sqrt(6) / sqrt(input_neurons + hidden_neurons);
W1 = 2*e_init_1*rand(input_neurons + 1,hidden_neurons) - e_init_1;

% W2 is a (X + 1) x 1 matrix - X + 1 hidden layer neurons, 1 output layer neuron
e_init_2 = sqrt(6) / sqrt(hidden_neurons + output_neurons);
W2 = 2*e_init_2*rand(hidden_neurons + 1,output_neurons) - e_init_1;

%%% 4. Repeat k times
%%% DON'T CHANGE
k = 150;

%%% 5. Some relevant variables
%%% DON'T CHANGE
m = size(X,1);
n = size(X,2);

%%% 6. Initialize cost array
%%% DON'T CHANGE
costs = zeros(k,1);

%%% 7. Set learning rate
%%% DON'T CHANGE
alpha = 5;

%%% 8. Implement Stochastic Gradient Descent
%%% PLACE YOUR CODE HERE
%Instructions seem to be implying that forward_propagation.m is never used.
%I think epoch is an iteration.
%As of writing this, epochs/iterations = 150.

%looping an equal amount of times as epochs/iterations
for aa = 1:1:1
    %looping through each example in x (m is equal to num of rows in x)
    for ab = 1:1:m
        %Get 1 example
        X0 = X(ab, :);
        %Transpose the example. Instructions want this and in-class
        %functions sorta want this. Strange that forwards_propagation.m did
        %not want to do this.
        X0 = transpose(X0);
        %Adding bias to X0
        X0 = [1; X0];
        disp(X0);
        %X0 Complete
    end
end

%%% 9. Plot the XOR points as well as the decision regions
%%% PLACE YOUR CODE HERE

%%% 10. Plot the cost per iteration
%%% PLACE YOUR CODE HERE