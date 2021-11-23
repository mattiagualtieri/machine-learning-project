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
[CCR, MAE, MMAE, tau] = ELM(X, Y);
printResults('S&P    ', CCR, MAE, MMAE, tau);
% Y = Moodys
Y = D(:,2);
[CCR, MAE, MMAE, tau] = ELM(X, Y);
printResults('Moodys', CCR, MAE, MMAE, tau);
% Y = Fitch
Y = D(:,3);
[CCR, MAE, MMAE, tau] = ELM(X, Y);
printResults('Fitch', CCR, MAE, MMAE, tau);

% --- functions ---

% Find optimal D function
function [Doptimal] = findOptimalD(X, Y, N, K)

    % split into train and test (25% test, 75% train)
    cv = cvpartition(N, 'HoldOut', 0.25);
    index = cv.test;
    Xtrain = X(~index,:);
    Ytrain = Y(~index,:);
    Xtest = X(index,:);
    Ytest = Y(index,:);
    Ntrain = cv.TrainSize;
    Ntest = cv.TestSize;
    
    % Find optimal hyperparameter D
    arrayCost = zeros(20, 2) + 50000;
    delta = 10e-3;
    i = 1;
    for D = 50:50:1000
        % Generate random W (K x D)
        W = rand(K, D)*2-1;
        % Calculate H = X * W (N x D)
        Htrain = 1./(1+(exp(-(Xtrain * W))));
        Htest = 1./(1+(exp(-(Xtest * W))));
        % Calculate Beta = (H'*H)^-1 * H'*Y (D x J)
        Beta = (inv((Htrain'*Htrain) +(delta * eye(size(Htrain, 2)))))*Htrain'*Ytrain;
        % Generate Y = H*Beta
        Ypredicted = Htest*Beta;
        % Calculate cost L
        L = norm(Ytest- Ypredicted);
        % every step we add arrayCost the row [L D]
        arrayCost(i, 1) = L;
        arrayCost(i, 2) = D;
        i = i + 1;
    end
    
    [~, indexMin] = min(arrayCost(:,1));
    Doptimal = arrayCost(indexMin,2);
end

% Regularized Extreme Learning Machine function
function [CCR, MAE, MMAE, tau] = ELM(X, Y)
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

    % split Y in Ytesting and Ytraining
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
    
    % D must be optimal
    D = findOptimalD(Xtrain, Ytrain, Ntrain, K);

    % Apply Extreme Learning Machine Algorithm
    delta = 10e-3;
    % Generate W (K x D)
    W = rand(K, D)*2-1;
    % Calculate H (N x D)
    Htrain = 1./(1+(exp(-(Xtrain * W))));
    Htest = 1./(1+(exp(-(Xtest * W))));
    % Generate Beta = (H'*H)^-1 * H'*Y  (D x J)
    Beta = (inv((Htrain'*Htrain) + (delta*eye(size(Htrain, 2)))))*Htrain'*Ytrain;
    % Calculate Y (N x J)
    Ypredicted = Htest*Beta;

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