%%% dsigmoid
%
% Function that computes the derivative of the sigmoid output of
% each element in the input.
% The input can be a scalar, vector or matrix (i.e. any numeric type).
%
% Inputs:
%  z - Input scalar, vector or matrix
%
% Outputs:
%  g - The same size as the input z which applies the derivative of the 
%  sigmoid function to each element individually.

function g = dsigmoid(z)
    g = sigmoid(z) .* ( 1 - sigmoid(z) );
end
%Different from what we learned.
%We learned (1 - g(z)^2)