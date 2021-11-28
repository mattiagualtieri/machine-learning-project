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
    % note: for the training set, we translate the only column of Y into a matrix
    % note: the matrix is (Ntrain x J)
    Ytrain = fromColumnToMatrix(Y(1:81,:));
    Ytest = Y(82:end,:);

    [Coptimal, Doptimal] = findOptimalHyperparameters(Xtrain, Ytrain);

    % Apply Extreme Learning Machine Algorithm
    W = rand(K, Doptimal)*2 - 1;
    bias_train_vector = rand(Ntrain, 1);
    bias_train = bias_train_vector(:,ones(1, Doptimal));
    bias_test_vector = rand(Ntest, 1);
    bias_test = bias_test_vector(:,ones(1, Doptimal));
    Htrain = 1 ./ (1 + (exp(-(Xtrain * W + bias_train))));
    Htest = 1 ./ (1 + (exp(-(Xtest * W + bias_test))));
    % Beta = [(H'*H + I/C)^-1] * (H' * Y)
    % note: Beta is (D x J)
    % note: eye(D) creates a diagonal identity matrix (D x D)
    Beta = (inv(((Htrain'*Htrain) + eye(Doptimal)/Coptimal))) * (Htrain'*Ytrain);
    Ypredicted = Htest * Beta;

    predicts = fromMatrixToColumn(Ypredicted);
    H = size(predicts, 1);
    
    % CCR --> correct cassification rate
    CCR = sum(predicts == Ytest)/H;
    % MAE --> mean absolute error
    MAE = sum(abs(predicts - Ytest))/H;
    % tau --> the Kendall's tau
    tau = corr(predicts, Ytest, 'type', 'Kendall');
    % MMAE --> maximum MAE
    MMAE = max(abs(predicts - Ytest));
    
end

% Find optimal hyperparameters function
function [C, D] = findOptimalHyperparameters(X, Y)

    [N, K] = size(X);
    % split into test and train (25% val, 75% trainval)
    cv = cvpartition(N, 'HoldOut', 0.25);
    index = cv.test;
    Xtrain = X(~index,:);
    Ytrain = Y(~index,:);
    Xtest = X(index,:);
    Ytest = Y(index,:);
    Ntrain = cv.TrainSize;
    Ntest = cv.TestSize;
    
    C = 10e-3;
    parametersMatrix = zeros(20, 2);
    i = 1;
    while C <= 10e3
        for D = 50:50:1000
            % Get W random matrix (K x D)
            W = rand(K, D)*2 - 1;
            % Bias (?)
            bias_train_vector = rand(Ntrain, 1);
            bias_train = bias_train_vector(:,ones(1, D));
            bias_test_vector = rand(Ntest, 1);
            bias_test = bias_test_vector(:,ones(1, D));
            % Compute H
            Htrain = 1 ./ (1 + (exp(-(Xtrain * W + bias_train))));
            Htest = 1 ./ (1 + (exp(-(Xtest * W + bias_test))));
            % Get Beta matrix (D x J)
            Beta = (inv((eye(D)/C)+(Htrain'*Htrain)))*(Htrain'*Ytrain);
            % Ypredicted = H * Beta
            Ypredicted = Htest * Beta;
            % Parameters matrix
            L = norm(Ytest - Ypredicted);
            % every step we add arrayCost the row [L D]
            parametersMatrix(i, 1) = C;
            parametersMatrix(i, 2) = D;
            parametersMatrix(i, 3) = L;
            i = i + 1;
        end
        C = C * 10;
    end

    % Find optimal C and D
    [~, index] = min(parametersMatrix(:,3));
    C = parametersMatrix(index, 1);
    D = parametersMatrix(index, 2);
end

% From column to matrix function
function [Y] = fromColumnToMatrix(Yoriginal)
    [N, ~] = size(Yoriginal);
    J = numel(unique(Yoriginal));
    Y = zeros(N, J);
    for i = 1:N
        column = Yoriginal(i);
        Y(i, column) = 1;
    end
end

function [Y] = fromMatrixToColumn(Yoriginal)
    H = size(Yoriginal, 1);
    Y = zeros(H, 1);
    for i = 1:H
        [~, position] = max(Yoriginal(i,:));
        Y(i) = position;
    end
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