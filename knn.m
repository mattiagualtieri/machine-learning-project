% load data from 'data.csv' file
D = load('data.csv');

% X matrix
X = D(:,4:end);
% Y contain only S&P column
Y = D(:,1);

% X scaling and normalization
Xs = scaleData(X);
Xn = normalizeData(Xs);

% split X in Xtest and Xtrain (10% test, 90% train)
cvx = cvpartition(size(Xn, 1), 'HoldOut', 0.1);
index = cvx.test;
Xtrain = Xn(~index,:);
Xtest = Xn(index,:);

% split Y in Ytest and Ytrain (10% test, 90% train)
cvy = cvpartition(size(Y, 1), 'HoldOut', 0.1);
index = cvy.test;
Ytrain = Y(~index,:);
Ytest = Y(index,:);

% split Xtrain into Xtrainval and Xval (25% val, 75% trainval)
cv = cvpartition(size(Xtrain, 1), 'HoldOut', 0.25);
index = cv.test;
Xtrainval = Xtrain(~index,:);
Xval = Xtrain(index,:);

% split Ytrain into Ytrainval and Yval (25% val, 75% trainval)
cv = cvpartition(size(Ytrain, 1), 'HoldOut', 0.25);
index = cv.test;
Ytrainval = Ytrain(~index,:);
Yval = Ytrain(index,:);

% starting the nesting cross validation
error = zeros(6, 1);
index = 1;

for k = 2:2:12
    [cIdx] = knnsearch(Xtrainval, Xval, 'K', k, 'Distance', 'euclidean');
    LabelsPerPattern = Ytrain(cIdx);
    Ypredicted = mode(LabelsPerPattern')';
    [H, ~] = size(Yval);
    C = sum(Ypredicted == Yval)/H;
    error(index) = C;
    index = index + 1;
end

[~, Kindex] = max(error);
K = (2*Kindex) - 1;

% execute k-NN with the optimal K value (K)
[cIdx] = knnsearch(Xtrain, Xtest, 'K', k, 'Distance', 'euclidean');
LabelsPerPattern = Ytrain(cIdx);
Ypredicted = mode(LabelsPerPattern')';
[H, ~] = size(Ytest);

% C --> correct cassification rate
C = sum(Ypredicted == Ytest)/H;
% MAE --> mean absolute error
MAE = sum(abs(Ypredicted - Ytest))/H;
% tau --> the Kendall's tau
tau = corr(Ypredicted, Ytest, 'type', 'Kendall');
% MMAE --> maximum MAE
MMAE = max(abs(Ypredicted - Ytest));

% --- functions ---

% Data scaling function
function [Xs] = scaleData(X)
    Xs = (X - min(X))./(max(X)-min(X));
end

% Data normalization function
function [Xn] = normalizeData(X)
    Xn = (X -mean(X))./(std(X));
end