
% ===========================================================  %

% KERNEL KNN Algorithm 
% Carlos Cuevas Baliñas
% Machine Learning - 4º IITV 

% Initial configuration 
clear all;
rng(1);
global sigma;

% Load dataset 
dataset = xlsread('BD_COUNTRY_RISK_EU.ods','BDTOTAL');

% Prepare data
[Xtrain,Ytrain1,Ytrain2,Ytrain3,Ntrain,Xtest,Ytest1,Ytest2,Ytest3,Ntest,N,J,K] = initData(dataset);

% Find optimal value of k neighbours and sigma parameter agency 1
k1 = findOptimalNeighbours(Xtrain,Ytrain1,Ntrain);
% Apply Kernel Knn Algorithm agency 1
[Ypredicted1,CCR1] = kernelKnnAlgorithm(Xtrain,Xtest,k1,Ytrain1,Ytest1,Ntest);
% Show results agency 1
disp("RATE AGENCY 1) S&P: ")
showResult(k1,CCR1);

% Find optimal value of k neighbours and sigma parameter agency 2
k2 = findOptimalNeighbours(Xtrain,Ytrain2,Ntrain);
% Apply Kernel Knn Algorithm agency 2
[Ypredicted2,CCR2] = kernelKnnAlgorithm(Xtrain,Xtest,k2,Ytrain2,Ytest2,Ntest);
% Show results agency 2
disp("RATE AGENCY 2) Moodys: ")
showResult(k2,CCR2);

% Find optimal value of k neighbours and sigma parameter agency 3
k3 = findOptimalNeighbours(Xtrain,Ytrain3,Ntrain);
% Apply Kernel Knn Algorithm agency 3
[Ypredicted3,CCR3] = kernelKnnAlgorithm(Xtrain,Xtest,k3,Ytrain3,Ytest3,Ntest);
% Show results agency 3
disp("RATE AGENCY 3) Fitch: ")
showResult(k3,CCR3);
% ===========================================================  %


% ===========================================================  %
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
% ===========================================================  %

% ===========================================================  %
function EK = euclideanKernel(Zi,Zj)

    % Zi -> Xtrain
    % Zj -> Xtest
    EK = zeros(size(Zj,1),1);
    for i=1:1:size(EK,1)
        EK(i,:) = 2 - 2*gaussianKernel(Zi,Zj(i,:));
    end
    
end
% ===========================================================  %

% ===========================================================  %
function GK = gaussianKernel(x,y)
    % Radial Basis Function 
    global sigma;
    aux1 = norm(x-y);
    aux2 = sigma * sigma;
    aux3 = -(aux1/aux2);
    GK = exp(aux3);
    
end
% ===========================================================  %

% ===========================================================  %
function [Ypredicted,CCR] = kernelKnnAlgorithm(Xtrain,Xtest,k,Ytrain,Ytest,Ntest)
    
    % Apply kernel knn algorithm
    [index] = knnsearch(Xtrain,Xtest,'K',k,'Distance',@euclideanKernel);
    
    % Calculate CCR
    LabelsKNN = Ytrain(index);
    Ypredicted = mode(LabelsKNN')'; 
    CCR = sum(Ypredicted == Ytest)/Ntest;
end
% ===========================================================  %

% ===========================================================  %
function showResult(k,CCR)
    global sigma;
    String1 = ['Optimal k neighbours: ',num2str(k)];
    String2 = ['Optimal sigma: ',num2str(sigma)];
    String3 = ['CCR Knn Algorithm: ',num2str(CCR)];
    disp(String1);
    disp(String2);
    disp(String3);
    disp(" ")
end
% ===========================================================  %

