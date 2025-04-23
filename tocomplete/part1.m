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
hidden_neurons = 8;

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
%for aa = 1:1:1
    %looping through each example in x (m is equal to num of rows in x)
%   for ab = 1:1:m
        %Get 1 example
%        X0 = X(ab, :);
        %Transpose the example. Instructions want this and in-class
        %functions sorta want this. Strange that forwards_propagation.m did
        %not want to do this.
%        X0 = transpose(X0);
        %Adding bias to X0
%        X0 = [1; X0];
%        disp(X0);
        %X0 Complete
%   end
%end

% To compute for the error in the hidden layer, the formula is:
% sigma^1 = (sigma^2 * Wno_bias^(2)) * g'(z^1)
% g'(z^1) is the derivative of the activation function or sigmoid
% Wno_bias^(2) is W2, first row removed (bias weight)
% Why use element-wise multiplication? Each node in the hidden layer has its own
% activation derivative, and backpropagated errors must be accounted for.
% For example: 
% delta3 = [0.1 0.4];       % Output layer error
% W2     = [0.5 0.3;        % W2: (hidden+1) x output
%          0.6 0.2;
%          0.1 0.9];       % 3 hidden neurons + bias
% z2 = [0.7 1.1 0.2];       % Pre-activation input into hidden layer
% Then:
% delta2 = (delta3 * W2(2:end,:)') .* dsigmoid(z2);
% Where delta3 * W2(2:end,:)' = total error signal backpropagated into each
% hidden neuron
% dsigmoid(z2) = sensitivity of each neuron at the time of activation
% .* element-wise combines both the backpropagated error scaled by the
% activity of the hidden neurons
% delta1 tells each hidden neuron how much correction is needed
for epoch = 1:k
    cost = 0;
    
    % Shuffle the training examples each epoch for optimal convergence
    shuffle = randperm(m);
    X = X(shuffle, :);
    y = y(shuffle, :);
    
    for idx = 1:m
        %%% Forward propagation for each example 
        x_single = X(idx, :);            % (1x2)
        y_single = y(idx);               % scalar (1x1)
        
        % Forward propagation data for a given epoch
        X0 = [1, x_single];              % (1x3), add bias
        S1 = X0 * W1;                    % (1x2)
        X1 = [1, sigmoid(S1)];           % (1x3), hidden layer with bias
        S2 = X1 * W2;                    % scalar
        X2 = sigmoid(S2);                % scalar, final prediction
        
        %%% Compute error (output - predicted)
        %delta2 = X2 - y_single;          % scalar
        %missing some of the formula
        delta2 = (X2 - y_single) .* dsigmoid(S2);
        
        %%% Backpropagate error to hidden layer
        delta1 = (delta2 * W2(2:end)') .* dsigmoid(S1); % (1x2)
        
        %%% Compute gradients for W1 and W2
        grad_W2 = (X1') * delta2;        % (3x1)
        grad_W1 = (X0') * delta1;        % (3x2)

        %%% Update weights (Gradient Descent)
        W1 = W1 - alpha * grad_W1;       % (3x2)
        W2 = W2 - alpha * grad_W2;       % (3x1)

        %%% Accumulate cost for this epoch
        %Formula that lab wants on page 7
        costs(epoch) = costs(epoch) + (1 / (2 * m) ) * norm( (X2 - y_single) );
        %cost = cost + ( -(y_single * log(X2) + (1 - y_single) * log(1 - X2)) );
    end
    
    % Average cost per example this epoch
    %costs(epoch) = cost / m;
end


%%% 9. Plot the XOR points as well as the decision regions
%%% PLACE YOUR CODE HERE
plot_XOR_and_regions(W1, W2);  % Plot the weights
title('XOR Decision Regions'); % Add a title to the plot

%%% 10. Plot the cost per iteration
%%% PLACE YOUR CODE HERE
figure; % Create a figure
plot(1:k, costs, 'b-', 'LineWidth', 2); % Plot the costs
xlabel('Epoch'); % x-axis label = Epoch
ylabel('Cost');  % y-axis label = Cost
title('Training Cost Over Each Epoch'); % Add a title to the plot