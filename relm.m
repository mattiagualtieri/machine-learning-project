% Initial configuration 
clear all;
rng(1);

% load data from 'data.csv' file
D = load('data.csv');

% X matrix
X = D(:,4:end);
% Y contain only S&P column
Y = D(:,1);

[N, K] = size(X);
J = numel(unique(Y));

% X scaling and normalization
Xs = scaleData(X);
Xn = normalizeData(Xs);

% split X in Xtest and Xtrain (25% test, 75% train)
cvx = cvpartition(size(Xn, 1), 'HoldOut', 0.25);
index = cvx.test;
Xtrain = Xn(~index,:);
Xtest = Xn(index,:);
[Ntrain, ~] = size(Xtrain);
[Ntest, ~] = size(Xtest);

% split Y in Ytest and Ytrain (25% test, 75% train)
cvy = cvpartition(size(Y, 1), 'HoldOut', 0.25);
index = cvy.test;
Ytraining = Y(~index,:);
Ytesting = Y(index,:);

% didn't get this step
Ytrain = zeros(Ntrain, J);
for i = 1:Ntrain
    column = Ytraining(i);
    Ytrain(i, column) = 1;
end

Ytest = zeros(Ntest, J);
for i = 1:Ntest
    column = Ytesting(i);
    Ytest(i, column) = 1;
end

% split into testval and trainval (25% val, 75% trainval)
cv = cvpartition(Ntrain, 'HoldOut', 0.25);
index = cv.test;
Xtrainval = Xtrain(~index,:);
Ytrainval = Ytrain(~index,:);
Xtestval = Xtrain(index,:);
Ytestval = Ytrain(index,:);
Ntrainval = cv.TrainSize;
Ntestval = cv.TestSize;

% Find optimal hyperparameters C and D
C = 10e-3;
parametersMatrix = [];

while C <= 10e3
    for D = 50:50:1000
        % Get W random matrix (K x D)
        W = rand(K, D)*2 - 1;
        % Bias
        bias_train_vector = rand(Ntrainval, 1);
        bias_train = bias_train_vector(:,ones(1, D));
        bias_test_vector = rand(Ntestval, 1);
        bias_test = bias_test_vector(:,ones(1, D));
        % Compute H
        Htrain = 1 ./ (1 + (exp(-(Xtrainval * W + bias_train))));
        Htest = 1 ./ (1 + (exp(-(Xtestval * W + bias_test))));
        % Get Beta matrix (D x J)
        Beta = (inv((eye(D)/C)+(Htrain'*Htrain)))*(Htrain'*Ytrainval);
        % Ypredicted = H * Beta
        Ypredicted = Htest * Beta;
        % Parameters matrix
        L = ((norm(Beta))^2) + (C*(norm((Htest * Beta) - Ypredicted))^2);
        row = [C D L];
        parametersMatrix = [parametersMatrix; row];
    end
    C = C*10;
end

% Find optimal C and D
[~, index] = min(parametersMatrix(:,3));
Coptimal = parametersMatrix(index, 1);
Doptimal = parametersMatrix(index, 2);

% Apply Extreme Learning Machine Algorithm
W = rand(K, Doptimal)*2 - 1;
bias_train_vector = rand(Ntrain, 1);
bias_train = bias_train_vector(:,ones(1, Doptimal));
bias_test_vector = rand(Ntest, 1);
bias_test = bias_test_vector(:,ones(1, Doptimal));
Htrain = 1 ./ (1 + (exp(-(Xtrain * W + bias_train))));
Htest = 1 ./ (1 + (exp(-(Xtest * W + bias_test))));
Beta = (inv((eye(Doptimal)/Coptimal)+(Htrain'*Htrain)))*(Htrain'*Ytrain);
Ypredicted = Htest * Beta;

% Calculate CCR
H = size(Ypredicted, 1);
predicts = zeros(H, 1);
for i = 1:size(Ypredicted, 1)
    [~, position] = max(Ypredicted(i,:));
    predicts(i) = position;
end

CCR = sum(predicts == Ytesting)/H;

% --- functions ---

% Data scaling function
function [Xs] = scaleData(X)
    Xs = (X - min(X))./(max(X)-min(X));
end

% Data normalization function
function [Xn] = normalizeData(X)
    Xn = (X -mean(X))./(std(X));
end