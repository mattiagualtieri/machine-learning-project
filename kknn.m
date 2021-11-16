% Initial configuration 
clear all;
rng(1);

global sigma

% load data from 'data.csv' file
D = load('data.csv');

% X matrix
X = D(:,4:end);
% Y contain only S&P column
Y = D(:,1);
[N, ~] = size(X);

% scaling and normalization of X
Xs = scaleData(X);
Xn = normalizeData(Xs);

% split X in Xtest and Xtrain (25% test, 75% train)
cvx = cvpartition(size(Xn, 1), 'HoldOut', 0.1);
index = cvx.test;
Xtrain = Xn(~index,:);
Xtest = Xn(index,:);

% split X in Xtest and Xtrain
% using 2007, 2008 and 2009 data for training
% using 2010 data for test
Xtrain = Xn(1:81,:);
Xtest = Xn(82:end,:);

% split Y in Ytest and Ytrain
% same here as X
Ytrain = Y(1:81,:);
Ytest = Y(82:end,:);

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
error = zeros(6, 7);
kValues = [1, 3, 5, 7, 9, 11];
sValues = [10^(-3), 10^(-2), 10^(-1), 10^(0), 10^(1), 10^(2), 10^(3)];
i = 1;

for k = kValues
    j = 1;
    for sigma = sValues
        [cIdx] = knnsearch(Xtrainval, Xval, 'K', k, 'Distance', @euclideanKernel);
        LabelsPerPattern = Ytrain(cIdx);
        Ypredicted = mode(LabelsPerPattern')';
        [H, ~] = size(Yval);
        CCR = sum(Ypredicted == Yval)/H;
        error(i, j) = CCR;
        j = j + 1;     
    end
    i = i + 1; 
end

[maxValue, ~] = max(error(:));
[i, j] = find(error == maxValue);
k = kValues(i(1));
sigma = sValues(j(1));

% execute Kernel K-NN with optimal hyperparameters
[cIdx] = knnsearch(Xtrain, Xtest, 'K', k, 'Distance', @euclideanKernel);
LabelsPerPattern = Ytrain(cIdx);
Ypredicted = mode(LabelsPerPattern')';
[H, ~] = size(Ytest);

% CCR --> correct cassification rate
CCR = sum(Ypredicted == Ytest)/H;
% MAE --> mean absolute error
MAE = sum(abs(Ypredicted - Ytest))/H;
% tau --> the Kendall's tau
tau = corr(Ypredicted, Ytest, 'type', 'Kendall');
% MMAE --> maximum MAE
MMAE = max(abs(Ypredicted - Ytest));

% --- functions ---

% Euclidean Kernel function
function [D2] = euclideanKernel(Zi, Zj)
    D2 = zeros(size(Zj, 1), 1);
    for i = 1:1:size(D2, 1)
       D2(i,:) =  2 - 2*gaussianKernel(Zi, Zj(i,:));
    end
end

% Gaussian Kernel function
function [dist] = gaussianKernel(X, Y)
    global sigma
    arg = (norm(X - Y)) / (sigma^2);
    dist = exp((-1)*arg);
end

% Data scaling function
function [Xs] = scaleData(X)
    Xs = (X - min(X))./(max(X)-min(X));
end

% Data normalization function
function [Xn] = normalizeData(X)
    Xn = (X -mean(X))./(std(X));
end