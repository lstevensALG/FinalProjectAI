%%% sigmoid
%
% Function that computes the sigmoid output of each element in the input.
% The input can be a scalar, vector or matrix (i.e. any numeric type).
%
% Inputs:
%  z - Input scalar, vector or matrix
%
% Outputs:
%  g - The same size as the input z which applies the sigmoid function to
%  each element individually.

function g = sigmoid(z)
    g = 1 ./ ( 1 + exp(-z) );
end