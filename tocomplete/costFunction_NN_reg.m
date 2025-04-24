%%% costFunction_nn_reg
%
% A cost function implementation of a neural network that has an input layer,
% one hidden layer and an output layer for the purposes of performing
% multi-class classification.  The goal is calculate what the cost
% would be to assign an input set of weights defined for each pair of nodes in
% every layer to this neural network architecture.  The function also returns
% the error gradient evaluated at this input set of weights.
% 
% Inputs:
%  X - A m x n matrix of training examples - m is the total number of examples
%  and n is the total number of features
%  y - A m x 1 column vector that determines output label for each training
%  example in X
%  lambda - The regularization parameter to prevent the neural network from
%  overfitting
%  input_neurons - The total number of input neurons at the input layer
%  hidden_neurons - The total number of hidden neurons at the hidden layer
%  output_neurons - The total number of output neurons at the output layer
%  weights - The neural network weights packed into a single vector
% 
% Outputs:
%  cost_val - The cost to assign the input set of weights to the neural network
%  grad - The gradient of the cost function evaluated for each weight given the
%  input weights
function [cost_val, grad] = costFunction_NN_reg(X, y, lambda, input_neurons, ...
                                  hidden_neurons, output_neurons, weights)

%%% 1. Compute the total amount of weights per layer
total_weights_W1 = (input_neurons + 1)*hidden_neurons;
total_weights_W2 = (hidden_neurons + 1)*output_neurons;

%%% 2. Extract out the right portions of the weights vector and reshape
W1 = reshape(weights(1:total_weights_W1), hidden_neurons, input_neurons + 1).';
W2 = reshape(weights(total_weights_W1+1:end), output_neurons, hidden_neurons + 1).';

%%% 3. Get number of training examples
m = size(X, 1);

%%% 4. Initialize total cost and gradient update matrices
cost_val = 0;
W1_update = zeros(size(W1));
W2_update = zeros(size(W2));

%%% 5. Compute total cost and gradient update matrices
%%% PLACE YOUR CODE HERE
X_bias = [ones(m, 1), X]; %  Add bias term

% Forward Propagation
z2 = X_bias * W1;
a2 = [ones(m, 1), sigmoid(z2)];
z3 = a2 * W2;
a3 = sigmoid(z3); % Final output (m x output_neurons)

% Convert labels to binary matrix form for multi-class
y_matrix = eye(output_neurons);
y_matrix = y_matrix(y, :);

% Compute regularized cost
J_unreg = (1/m) * sum(sum(-y_matrix.*log(a3)-(1-y_matrix).*log(1-a3)));
regularization = (lambda/(2*m)) * (sum(sum(W1(2:end,:).^2)) + sum(sum(W2(2:end,:).^2)));
cost_val = J_unreg + regularization;

% Backpropagation
delta3 = a3 - y_matrix;
delta2 = (delta3 * W2(:, 2:end)).*dsigmoid(z2);

% Update weights
W2_update = (1/m) * (a2' * delta3) + (lambda/m)*[zeros(1,size(W2,2)); W2(2:end,:)];
W1_update = (1/m) * (X_bias' * delta2) + (lambda/m)*[zeros(1,size(W1,2)); W1(2:end,:)];

%%% 6. Take the updates and pack the output parameter vector
grad = zeros(numel(weights),1);
grad(1:total_weights_W1) = reshape(W1_update.', total_weights_W1, 1);
grad(total_weights_W1+1:end) = reshape(W2_update.', total_weights_W2, 1);