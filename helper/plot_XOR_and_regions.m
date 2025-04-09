function plot_XOR_and_regions(W1, W2)

%%% 1. Define inputs
X = [0 1; 1 1; 1 0; 0 0];

%%% 2. Define a set of coordinates for each feature between 0 and 1 in 
%%% steps of 0.001.  
[X1,X2] = meshgrid(0:0.001:1, 0:0.001:1);

%%% 3. Convert into column vectors and create a new input matrix that stacks these
%%% columns together
X1 = X1(:);
X2 = X2(:);
XVALS = [X1 X2];

%%% 4. Compute raw outputs from the output layer
%%% NOTE: This requires that forward_propagation be completed successfully
h = forward_propagation(XVALS, W1, W2);

%%% 5. Predict which class each input belongs to
Y = h >= 0.5;

%%% 6. Spawn new figure
figure; hold on;

%%% 7. Plot the actual training examples
% y = 0 are red crosses
% y = 1 are blue circles
plot([X(1,1) X(3,1)], [X(1,2) X(3,2)], 'rx', 'MarkerSize', 16);
plot([X(2,1) X(4,1)], [X(2,2) X(4,2)], 'bo', 'MarkerSize', 16);

%%% 8. Plot decision regions
gscatter(X1, X2, Y, [0.85 0.325 0.098; 0.9290 0.6940 0.1250]);
axis tight;