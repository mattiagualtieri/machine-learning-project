% ===========================================================  %

% KERNEL KNN ALGORITHM
clear all;
rng(1);

% Load dataset 
dataset = xlsread('BD_COUNTRY_RISK_EU.ods','BDTOTAL');

dataX = dataset(:,4:end);
dataY = dataset(:,1:3);

J = numel(unique(dataY));
[N,K] = size(dataX);
% Initial configuration 
global sigma;
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
k1 = findOptimalNeighbours1(Xtrain,Ytrain1,Ntrain);
k2 = findOptimalNeighbours1(Xtrain,Ytrain2,Ntrain);
k3 = findOptimalNeighbours1(Xtrain,Ytrain3,Ntrain);

% Apply knn algorithm 
[Ypredicted1,CCR1] = knnKernelAlgorithm(Xtrain,Xtest,k1,Ytrain1,Ytest1,Ntest);
[Ypredicted2,CCR2] = knnKernelAlgorithm(Xtrain,Xtest,k2,Ytrain2,Ytest2,Ntest);
[Ypredicted3,CCR3] = knnKernelAlgorithm(Xtrain,Xtest,k3,Ytrain3,Ytest3,Ntest);

Ypredichas = [Ypredicted1 Ypredicted2 Ypredicted3]
% Show results
disp("RATE AGENCY 1) S&P: ")
showResult(k1,CCR1);
disp("RATE AGENCY 2) Moodys: ")
showResult(k2,CCR2);
disp("RATE AGENCY 3) Fitch: ")
showResult(k3,CCR3);


function k_optimal = findOptimalNeighbours1(Xtrain,Ytrain,Ntrain)

global sigma;


CVHoldOut = cvpartition(Ntrain,'HoldOut',0.25);%'Stratify',false
    
    NtrainVal = CVHoldOut.TrainSize; %#ok
    NtestVal = CVHoldOut.TestSize;
    
    XtrainVal = Xtrain(CVHoldOut.training(),:);
    YtrainVal = Ytrain(CVHoldOut.training(),:);
    
    XtestVal = Xtrain(CVHoldOut.test(),:);
    YtestVal = Ytrain(CVHoldOut.test(),:);


%k-NN Kernel

%Matriz de errores
error = zeros(6,7);
%vector del hiperparametro K
K = [1 3 5 7 9 11];


sigma = 0.001;
mejorCCR = 0;

sigma_optimal = sigma;
for i = 1:1:6   
    for j = 1:1:7
        [Neighbours] = knnsearch(XtrainVal, XtestVal, 'K',K(i),'Distance',@EK);
        LabelsPerPattern = YtrainVal(Neighbours);
        LabelsKNN = mode(LabelsPerPattern')';
        ccrKNN = sum(LabelsKNN == YtestVal)/NtestVal;
        error(i,j) = ccrKNN; 

        if (ccrKNN > mejorCCR) %si el CCR actual es mejor que el anterior, se almacena y se actualiza la K y la sigma óptimas
            mejorCCR = ccrKNN;
            k_optimal = K(i);
            sigma_optimal = sigma;
            
        end
        
    end
    sigma = sigma * 10;
end

[~, k_optimal] = max(max(error'));
k_optimal = (k_optimal * 2) - 1;

sigma = sigma_optimal;
end

function [Ypredicted,CCR] = knnKernelAlgorithm(Xtrain,Xtest,k_optimal,Ytrain,Ytest,Ntest)
    
[Neighbours] = knnsearch(Xtrain,Xtest,'K',k_optimal, 'Distance', @EK);

LabelsPerPattern = Ytrain(Neighbours);
Ypredicted = mode(LabelsPerPattern')';
CCR = sum(Ypredicted == Ytest)/Ntest;

end
function EuclideanKernel = EK(Zi,Zj)
    EuclideanKernel = zeros(size(Zj,1),1);
    for i=1:1:size(EuclideanKernel,1)
        EuclideanKernel(i,:) = 2-2*GK(Zi,Zj(i,:));
    end
end

function GaussianKernel = GK(x,y)
    Norma = norm(x-y); 
    global sigma;
    t = -Norma * (sigma*sigma);
    GaussianKernel = exp(t);
    
end
function showResult(k,CCR)
    String1 = ['Optimal k neighbours: ',num2str(k)];
    String2 = ['CCR Knn Algorithm: ',num2str(CCR)];
    disp(String1);
    disp(String2);
    disp(" ")
end
