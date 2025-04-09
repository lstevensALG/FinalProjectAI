%%% 1. Load in the data
close all;
clearvars;
load fisheriris;

%%% 2. Get only the last two features, but we want all of the data
X = meas(1:150,3:4);

%%% 3. Show data first
figure;
plot(X(1:50,1), X(1:50,2), 'b.', X(51:100,1), X(51:100,2), 'r.',...
    X(101:150,1), X(101:150,2), 'g.', 'MarkerSize', 16);

%%% 4. Train SVMs
svms = cell(1,3); % Stores SVM model for detecting each class
% Cell arrays are flexible arrays where each element can store ANYTHING

% Train first SVM
y1 = false(150,1);
y1(1:50) = true;
svms{1} = fitcsvm(X, y1, 'ClassNames', [false true]);

% Train second SVM
y2 = false(150,1);
y2(51:100) = true;
svms{2} = fitcsvm(X, y2, 'ClassNames', [false true]);

% Train third SVM
y3 = false(150,1);
y3(101:150) = true;
svms{3} = fitcsvm(X, y3, 'ClassNames', [false true]);

%%% 5. Use one-vs-all to predict new instances
newX = [1.7 2; 4.5 1.2; 6 2];

% For each SVM, calculate the matching score for each set of new examples
scores = zeros(size(newX,1), 3);

for ii = 1 : 3
    % This version of predict has a second output argument where
    % it provides a two column matrix of matching scores
    % The first column is the score for the negative class
    % The second column is the score for the positive class
    [~,score] = predict(svms{ii}, newX);

    % Place positive class scores in the right column
    scores(:,ii) = score(:,2);
end

%%% 6. Find the labels
[~,labels] = max(scores,[],2);
