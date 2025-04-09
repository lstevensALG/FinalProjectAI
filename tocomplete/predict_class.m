%%% predict_class
%
% Function that takes in a score matrix and outputs the predicted class
% for each example in the score matrix
%
% Inputs:
%  Y - Score matrix of size m x N - m is the total number of examples and 
%  N is the total number of classes
%
% Outputs:
%  classes - A m x 1 column vector that determines which class each input
%  example belongs to.  Note that the classes are enumerated from 1 up to N 
%  this time around isntead of 0 to N - 1 as seen in Lab #2
function classes = predict_class(Y)
%%% PLACE YOUR CODE HERE