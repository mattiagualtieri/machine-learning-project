% Initial configuration 
clearvars;
clc;
rng(1);

% load data from 'data.csv' file
D = load('data.csv');

fprintf('\t\t\tCCR\t\t\tMAE\t\t\tMMAE\t\ttau\n');

% X matrix
X = D(:,4:end);
% Y = S&P
Y = D(:,1);
[CCR, MAE, MMAE, tau] = RELM(X, Y);
printResults('S&P    ', CCR, MAE, MMAE, tau);
% Y = Moodys
Y = D(:,2);
[CCR, MAE, MMAE, tau] = RELM(X, Y);
printResults('Moodys', CCR, MAE, MMAE, tau);
% Y = Fitch
Y = D(:,3);
[CCR, MAE, MMAE, tau] = RELM(X, Y);
printResults('Fitch', CCR, MAE, MMAE, tau);

% --- functions ---

% Regularized Extreme Learning Machine function
function [CCR, MAE, MMAE, tau] = RELM(X, Y)
    [~, K] = size(X);
    J = numel(unique(Y));

    % X scaling and normalization
    Xs = scaleData(X);
    Xn = normalizeData(Xs);

    % split X in Xtest and Xtrain
    % using 2007, 2008 and 2009 data for training
    % using 2010 data for test
    Xtrain = Xn(1:81,:);
    Xtest = Xn(82:end,:);
    [Ntrain, ~] = size(Xtrain);
    [Ntest, ~] = size(Xtest);

    % split Y in Ytest and Ytrain
    % same here as X
    Ytraining = Y(1:81,:);
    Ytesting = Y(82:end,:);

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
            L = norm(Ytestval - Ypredicted);
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

    H = size(Ypredicted, 1);
    predicts = zeros(H, 1);
    for i = 1:size(Ypredicted, 1)
        [~, position] = max(Ypredicted(i,:));
        predicts(i) = position;
    end
    
    % CCR --> correct cassification rate
    CCR = sum(predicts == Ytesting)/H;
    % MAE --> mean absolute error
    MAE = sum(abs(predicts - Ytesting))/H;
    % tau --> the Kendall's tau
    tau = corr(predicts, Ytesting, 'type', 'Kendall');
    % MMAE --> maximum MAE
    MMAE = max(abs(predicts - Ytesting));
    
end

% Data scaling function
function [Xs] = scaleData(X)
    Xs = (X - min(X))./(max(X)-min(X));
end

% Data normalization function
function [Xn] = normalizeData(X)
    Xn = (X -mean(X))./(std(X));
end

% Print results function
function printResults(name, CCR, MAE, MMAE, tau)
    fprintf('%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n', name, CCR, MAE, MMAE, tau);
end