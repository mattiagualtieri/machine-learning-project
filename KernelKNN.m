% ======================================================================= %

% MACHINE LEARNING PROJECT - MACHINE LEARNING 4ºIITV
% Group 1: Addressing the EU Sovereign Ratings
% Álvaro Bersabé, Carlos Cuevas, Mattia Gualtieri, Álvaro Jiménez

% Initial configuration 
clear all;
rng(1);
global sigma;

% Load dataset 
dataset = xlsread('BD_COUNTRY_RISK_EU.ods','BDTOTAL');

% Prepare data
[Xtrain,Ytrain1,Ytrain2,Ytrain3,Ntrain,Xtest,Ytest1,Ytest2,Ytest3,Ntest,N,J,K] = initData(dataset);

% Select Rating Agency 
agency = 3;  % 1 = S&P //  2 = Moodys // 3 = Fitch

% Knn for agency 1
if agency == 1
    [Ypredicted,CCR,k] = kernelKNN(Xtrain,Xtest,Ytrain1,Ytest1,Ntrain,Ntest);
    disp("=======================================================================");
    disp("RATE AGENCY 1) S&P: ")
    showResult(k,CCR);
    disp("=======================================================================");
end
% Knn for agency 2
if agency == 2
    [Ypredicted,CCR,k] = kernelKNN(Xtrain,Xtest,Ytrain2,Ytest2,Ntrain,Ntest);
    disp("=======================================================================");
    disp("RATE AGENCY 2) Moodys: ")
    showResult(k,CCR);
    disp("=======================================================================");
end
% Knn for agency 3
if agency == 3
    [Ypredicted,CCR,k] = kernelKNN(Xtrain,Xtest,Ytrain3,Ytest3,Ntrain,Ntest);
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
function [Ypredicted,CCR,k] = kernelKNN(Xtrain,Xtest,Ytrain,Ytest,Ntrain,Ntest)
    % Find optimal value of k neighbours
    k = findOptimalNeighbours(Xtrain,Ytrain,Ntrain);
    % Apply Kernel Knn Algorithm
    [Ypredicted,CCR] = kernelKnnAlgorithm(Xtrain,Xtest,k,Ytrain,Ytest,Ntest);
end
% ======================================================================= %


% ======================================================================= %
function k_optimal = findOptimalNeighbours(Xtrain,Ytrain,Ntrain)
    global sigma;
    % Split data into trainValidation and testValidation
    CVHoldOut = cvpartition(Ntrain,'HoldOut',0.25,'Stratify',false);
    
    NtrainVal = CVHoldOut.TrainSize; %#ok
    NtestVal = CVHoldOut.TestSize;
    
    XtrainVal = Xtrain(CVHoldOut.training(),:);
    YtrainVal = Ytrain(CVHoldOut.training(),:);
    
    XtestVal = Xtrain(CVHoldOut.test(),:);
    YtestVal = Ytrain(CVHoldOut.test(),:);
    
    % Find optimal value of k and sigma
    sigma = 10e-3;
    arrayCCR = [];
    while sigma <= 10e3
        for k=1:2:13
            [index] = knnsearch(XtrainVal,XtestVal,'K',k,'Distance',@euclideanKernel);
            LabelsKNN = YtrainVal(index);
            Ypredicted = mode(LabelsKNN')';
            CCR = sum(Ypredicted == YtestVal)/NtestVal;
            row = [CCR k sigma];
            arrayCCR = [arrayCCR; row]; %#ok
        end
        sigma = sigma*10;
    end
    [~,indexMax] = max(arrayCCR(:,1));
    k_optimal = arrayCCR(indexMax,2);
    sigma_optimal = arrayCCR(indexMax,3);
    sigma = sigma_optimal;
end
% ======================================================================= %

% ======================================================================= %
function EK = euclideanKernel(Zi,Zj)

    % Zi -> Xtrain
    % Zj -> Xtest
    EK = zeros(size(Zj,1),1);
    for i=1:1:size(EK,1)
        EK(i,:) = 2 - 2*gaussianKernel(Zi,Zj(i,:));
    end
    
end
% ======================================================================= %

% ======================================================================= %
function GK = gaussianKernel(x,y)
    % Radial Basis Function 
    global sigma;
    aux1 = norm(x-y);
    aux2 = sigma * sigma;
    aux3 = -(aux1/aux2);
    GK = exp(aux3);
    
end
% ======================================================================= %

% ======================================================================= %
function [Ypredicted,CCR] = kernelKnnAlgorithm(Xtrain,Xtest,k,Ytrain,Ytest,Ntest)
    
    % Apply kernel knn algorithm
    [index] = knnsearch(Xtrain,Xtest,'K',k,'Distance',@euclideanKernel);
    
    % Calculate CCR
    LabelsKNN = Ytrain(index);
    Ypredicted = mode(LabelsKNN')'; 
    CCR = sum(Ypredicted == Ytest)/Ntest;
end
% ======================================================================= %

% ======================================================================= %
function showResult(k,CCR)
    global sigma;
    String1 = ['Optimal k neighbours: ',num2str(k)];
    String2 = ['Optimal sigma: ',num2str(sigma)];
    String3 = ['CCR Knn Algorithm: ',num2str(CCR)];
    disp(String1);
    disp(String2);
    disp(String3);
end
% ======================================================================= %

