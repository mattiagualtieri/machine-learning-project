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
[CCR, MAE, MMAE, tau] = KNN(X, Y);
printResults('S&P    ', CCR, MAE, MMAE, tau);
% Y = Moodys
Y = D(:,2);
[CCR, MAE, MMAE, tau] = KNN(X, Y);
printResults('Moodys', CCR, MAE, MMAE, tau);
% Y = Fitch
Y = D(:,3);
[CCR, MAE, MMAE, tau] = KNN(X, Y);
printResults('Fitch', CCR, MAE, MMAE, tau);

% --- functions ---

% K-NN function
function [CCR, MAE, MMAE, tau] = KNN(X, Y)

    % X scaling and normalization
    Xs = scaleData(X);
    Xn = normalizeData(Xs);

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
    error = zeros(6, 1);
    index = 1;

    for k = 2:2:12
        [cIdx] = knnsearch(Xtrainval, Xval, 'K', k, 'Distance', 'euclidean');
        LabelsPerPattern = Ytrain(cIdx);
        Ypredicted = mode(LabelsPerPattern')';
        [H, ~] = size(Yval);
        CCR = sum(Ypredicted == Yval)/H;
        error(index) = CCR;
        index = index + 1;
    end

    [~, Kindex] = max(error);
    K = (2*Kindex) - 1;

    % execute k-NN with the optimal K value (K)
    [cIdx] = knnsearch(Xtrain, Xtest, 'K', K, 'Distance', 'euclidean');
    LabelsPerPattern = Ytrain(cIdx);
    Ypredicted = mode(LabelsPerPattern')';
    [H, ~] = size(Ytest);

    % CCR --> correct cassification rate
    CCR = sum(Ypredicted == Ytest)/H;
    % MAE --> mean absolute error
    MAE = sum(abs(Ypredicted - Ytest))/H;
    % tau --> the Kendall's tau
    tau = 0;%corr(Ypredicted, Ytest, 'type', 'Kendall');
    % MMAE --> maximum MAE
    MMAE = max(abs(Ypredicted - Ytest));
    
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