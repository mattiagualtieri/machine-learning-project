% load data from 'datos.csv' file
D = load('datos.csv');

% X matrix
X = D(:,4:end);
% Y contains only S&P column
Y = D(:,1);

% X scaling and normalization
Xs = scaleData(X);
Xn = normalizeData(Xs);

% split X in Xtest and Xtrain (25% test, 75% train)
cvx = cvpartition(size(Xn, 1), 'HoldOut', 0.25);
index = cvx.test;
Xtrain = Xn(~index,:);
Xtest = Xn(index,:);

% split Y in Ytest and Ytrain (25% test, 75% train)
cvy = cvpartition(size(Y, 1), 'HoldOut', 0.25);
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
    LabelsKNN = mode(LabelsPerPattern')';
    [H, ~] = size(Yval);
    ccr = sum(LabelsKNN == Yval)/H;
    error(index) = ccr;
    index = index + 1;
end

[~, Kindex] = max(error);
K = (2*Kindex) - 1;

% execute k-NN with the optimal K value (K)
[cIdx] = knnsearch(Xtrain, Xtest, 'K', k, 'Distance', 'euclidean');
LabelsPerPattern = Ytrain(cIdx);
LabelsKNN = mode(LabelsPerPattern')';
[H, ~] = size(Ytest);
ccr = sum(LabelsKNN == Ytest)/H;

% --- functions ---

% Data scaling function
function [Xs] = scaleData(X)
    Xs = (X - min(X))./(max(X)-min(X));
end

% Data normalization function
function [Xn] = normalizeData(X)
    Xn = (X -mean(X))./(std(X));
end