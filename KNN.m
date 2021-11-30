% ======================================================================= %

% MACHINE LEARNING PROJECT - MACHINE LEARNING 4ºIITV
% Group 1: Addressing the EU Sovereign Ratings
% Álvaro Bersabé, Carlos Cuevas, Mattia Gualtieri, Álvaro Jiménez

% Initial configuration 
clear all;
rng(1);

% Load dataset 
dataset = xlsread('BD_COUNTRY_RISK_EU.ods','BDTOTAL');

% Prepare data
[Xtrain,Ytrain1,Ytrain2,Ytrain3,Ntrain,Xtest,Ytest1,Ytest2,Ytest3,Ntest,N,J,K] = initData(dataset);

% Select Rating Agency 
agency = 3;  % 1 = S&P //  2 = Moodys // 3 = Fitch

% Knn for agency 1
if agency == 1
    [Ypredicted,CCR,k] = knn(Xtrain,Xtest,Ytrain1,Ytest1,Ntrain,Ntest);
    disp("=======================================================================");
    disp("RATE AGENCY 1) S&P: ")
    showResult(k,CCR);
    disp("=======================================================================");
end
% Knn for agency 2
if agency == 2
    [Ypredicted,CCR,k] = knn(Xtrain,Xtest,Ytrain2,Ytest2,Ntrain,Ntest);
    disp("=======================================================================");
    disp("RATE AGENCY 2) Moodys: ")
    showResult(k,CCR);
    disp("=======================================================================");
end
% Knn for agency 3
if agency == 3
    [Ypredicted,CCR,k] = knn(Xtrain,Xtest,Ytrain3,Ytest3,Ntrain,Ntest);
    disp("=======================================================================");
    disp("RATE AGENCY 3) Fitch: ")
    showResult(k,CCR);
    disp("=======================================================================");
end
% Error
if agency < 1 || agency > 3
    disp("Error selecting agency");
end

% ======================================================================= %

% ======================================================================= %
function [Ypredicted,CCR,k] = knn(Xtrain,Xtest,Ytrain,Ytest,Ntrain,Ntest)
    % Get optimal value of k neighbours
    k = findOptimalNeighbours(Xtrain,Ytrain,Ntrain);
    % Apply knn algorithm
    [Ypredicted,CCR] = knnAlgorithm(Xtrain,Xtest,k,Ytrain,Ytest,Ntest);
end
% ======================================================================= %

% ======================================================================= %
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
% ======================================================================= %

% ======================================================================= %
function [Ypredicted,CCR] = knnAlgorithm(Xtrain,Xtest,k,Ytrain,Ytest,Ntest)
    
    % Apply knn algorithm
    [index] = knnsearch(Xtrain,Xtest,'K',k,'Distance','Euclidean');
    
    % Calculate CCR
    LabelsKNN = Ytrain(index);
    Ypredicted = mode(LabelsKNN')';
    CCR = sum(Ypredicted == Ytest)/Ntest;
   
end
% ======================================================================= %

% ======================================================================= %
function showResult(k,CCR)
    String1 = ['Optimal k neighbours: ',num2str(k)];
    String2 = ['CCR Knn Algorithm: ',num2str(CCR)];
    disp(String1);
    disp(String2);
end
% ======================================================================= %