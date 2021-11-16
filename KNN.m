% ===========================================================  %

% KNN Algorithm 
% Carlos Cuevas Baliñas
% Machine Learning - 4º IITV 

% Initial configuration 
clear all;
rng(1);

% Load dataset 
dataset = xlsread('BD_COUNTRY_RISK_EU.ods','BDTOTAL');

dataX = dataset(:,4:end);
dataY = dataset(:,1:3);

J = numel(unique(dataY));
[N,K] = size(dataX);

% Data normalization
dataX = (dataX - min(dataX))./(max(dataX) - min(dataX));

% Split data into Train and Test
Xtrain = dataX(1:81,:);
Ytrain1 = dataY(1:81,1);
Ytrain2 = dataY(1:81,2);
Ytrain3 = dataY(1:81,3);
Ntrain = size(Xtrain,1);

Xtest = dataX(82:end,:);
Ytest1 = dataY(82:end,1);
Ytest2 = dataY(82:end,2);
Ytest3 = dataY(82:end,3);
Ntest = size(Xtest,1);

% Get optimal value of k neighbours
k1 = findOptimalNeighbours(Xtrain,Ytrain1,Ntrain);
k2 = findOptimalNeighbours(Xtrain,Ytrain2,Ntrain);
k3 = findOptimalNeighbours(Xtrain,Ytrain3,Ntrain);

% Apply knn algorithm 
[Ypredicted1,CCR1] = knnAlgorithm(Xtrain,Xtest,k1,Ytrain1,Ytest1,Ntest);
[Ypredicted2,CCR2] = knnAlgorithm(Xtrain,Xtest,k2,Ytrain2,Ytest2,Ntest);
[Ypredicted3,CCR3] = knnAlgorithm(Xtrain,Xtest,k3,Ytrain3,Ytest3,Ntest);

% Show results
disp("RATE AGENCY 1) S&P: ")
showResult(k1,CCR1);
disp("RATE AGENCY 2) Moodys: ")
showResult(k2,CCR2);
disp("RATE AGENCY 3) Fitch: ")
showResult(k3,CCR3);

% ===========================================================  %

% ===========================================================  %
function k_optimal = findOptimalNeighbours(Xtrain,Ytrain,Ntrain)
    
    % Split into trainValidation and testValidation
    CVHoldOut = cvpartition(Ntrain,'HoldOut',0.25,'Stratify',false);
    
    NtrainVal = CVHoldOut.TrainSize; %#ok
    NtestVal = CVHoldOut.TestSize;
    
    XtrainVal = Xtrain(CVHoldOut.training(),:);
    YtrainVal = Ytrain(CVHoldOut.training(),:);
    
    XtestVal = Xtrain(CVHoldOut.test(),:);
    YtestVal = Ytrain(CVHoldOut.test(),:);
    
    % Find optimal value of k
    arrayCCR = [];
    
    for k=1:2:13
        [index] = knnsearch(XtrainVal,XtestVal,'K',k,'Distance','Euclidean');
        LabelsKNN = YtrainVal(index);
        Ypredicted = mode(LabelsKNN')';
        CCR = sum(Ypredicted == YtestVal)/NtestVal;
        arrayCCR = [arrayCCR; CCR]; %#ok
    end
    
    [~, indexMax] = max(arrayCCR'); %#ok
    k_optimal = (indexMax * 2) - 1;
    
end
% ===========================================================  %

% ===========================================================  %
function [Ypredicted,CCR] = knnAlgorithm(Xtrain,Xtest,k,Ytrain,Ytest,Ntest)
    
    % Apply knn algorithm
    [index] = knnsearch(Xtrain,Xtest,'K',k,'Distance','Euclidean');
    
    % Calculate CCR
    LabelsKNN = Ytrain(index);
    Ypredicted = mode(LabelsKNN')';
    CCR = sum(Ypredicted == Ytest)/Ntest;
   
end
% ===========================================================  %

% ===========================================================  %
function showResult(k,CCR)
    String1 = ['Optimal k neighbours: ',num2str(k)];
    String2 = ['CCR Knn Algorithm: ',num2str(CCR)];
    disp(String1);
    disp(String2);
    disp(" ")
end
% ===========================================================  %


