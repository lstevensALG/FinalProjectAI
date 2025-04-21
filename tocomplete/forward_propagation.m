%%% forward_propagation
%
% A function that takes in a matrix of examples, as well as the weight matrices
% defined between the input layer and hidden layer, and between 
% the hidden layer and the output layer to compute the RAW predictions at 
% the output layer.  The output layer is not thresholded here, nor does any
% One-Vs-All action happen here.
%
% Inputs:
%  X - The input matrix of examples of size m x d0. m is the total number of
%  examples and d0 is the total number of neurons at the input layer without 
%  the bias unit, or the total number of features
%  W1 - The weight matrix defined between the input layer and hidden layer
%  of size (d0 + 1) x d1. d1 is the total number of neurons at the hidden layer
%  without the bias unit.
%  W2 - The weight matrix defined between the hidden layer and output layer
%  of size (d1 + 1) x d2.  d2 is the total number of neurons at the output
%  layer.
%
% Outputs:
%  Y - A m x d2 matrix that stores the raw predictions of the output layer.
%  Each row stores the raw outputs while each column represents one neuron from
%  the output layer.
function Y = forward_propagation(X, W1, W2)
    
    % Get the total number of examples
    m = size(X,1);
    
%%%% PLACE CODE HERE
%   First, we have to calculate X0
%   We have to add a column of ones to X for bias.
%   First column: Column of ones.
%   Other columns: Whatever the hell has been put in X.
%   We don't have to transpose like in class because I think input X is on
%   its side (already transposed?)
    X0 = [ ones(m, 1), X ];

%   Now, we have X0. We continue on by calculating S1.
%   We just multiply X0 with W1.
    S1 = mtimes(X0, W1);

%   Now, we have to calculate X1 from S1.
%   This requires us to sigmoid function S1,
%   Then, add a column of ones to act as a bias.
    X1 = sigmoid(S1);
    X1 = [ ones(m, 1), X1 ];

%   Calculate S2 with X1
    S2 = mtimes(X1, W2);

%   Calculate X2 from S2.
%   This time, we just sigmoid function S2 and don't add bias because it's
%   the last layer.
    X2 = sigmoid(S2);

%   We now have the final layer neuron results. Just assign X2 to Y for
%   output.
    Y = X2;
end
