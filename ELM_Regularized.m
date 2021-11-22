% ===========================================================  %

% Extreme Learning Machine Regularized Algorithm 
% Carlos Cuevas Baliñas
% Machine Learning - 4º IITV 

% Initial configuration 
clear all;
rng(1);

% Load dataset 
dataset = xlsread('BD_COUNTRY_RISK_EU.ods','BDTOTAL');

% Prepare data
[Xtrain,Ytrain1,Ytrain2,Ytrain3,Ntrain,Xtest,Ytest1,Ytest2,Ytest3,Ntest,N,J,K] = initData(dataset);

% 1 of J 
Ytrain1 = generate1ofJLabel(Ytrain1,Ntrain,J);
Ytrain2 = generate1ofJLabel(Ytrain2,Ntrain,J);
Ytrain3 = generate1ofJLabel(Ytrain3,Ntrain,J);

Ytest1 = generate1ofJLabel(Ytest1,Ntest,J);
Ytest2 = generate1ofJLabel(Ytest2,Ntest,J);
Ytest3 = generate1ofJLabel(Ytest3,Ntest,J);

% Get optimal value of C and D agency 1
[C_optimal1,D_optimal1] = findOptimalHyperparameters(Xtrain,Ytrain1,Ntrain,K);
% Apply Extreme Learning Machine agency 1
[Ypredicted1,CCR1,CCR_ELM1] = extremeLearningMachine(Xtrain,Ytrain1,Xtest,Ytest1,Ntrain,Ntest,K,C_optimal1,D_optimal1);
% Show results agency 1
disp("RATE AGENCY 1) S&P: ")
showResult(C_optimal1,D_optimal1,CCR1,CCR_ELM1);

% Get optimal value of C and D agency 2
[C_optimal2,D_optimal2] = findOptimalHyperparameters(Xtrain,Ytrain1,Ntrain,K);
% Apply Extreme Learning Machine agency 2
[Ypredicted2,CCR2,CCR_ELM2] = extremeLearningMachine(Xtrain,Ytrain2,Xtest,Ytest2,Ntrain,Ntest,K,C_optimal2,D_optimal2);
% Show results agency 2
disp("RATE AGENCY 2) Moodys: ")
showResult(C_optimal2,D_optimal2,CCR2,CCR_ELM2);

% Get optimal value of C and D agency 3
[C_optimal3,D_optimal3] = findOptimalHyperparameters(Xtrain,Ytrain1,Ntrain,K);
% Apply Extreme Learning Machine agency 3
[Ypredicted3,CCR3,CCR_ELM3] = extremeLearningMachine(Xtrain,Ytrain3,Xtest,Ytest3,Ntrain,Ntest,K,C_optimal3,D_optimal3);
% Show results agency 3
disp("RATE AGENCY 3) Fitch: ")
showResult(C_optimal3,D_optimal3,CCR3,CCR_ELM3);

% ===========================================================  %

% ===========================================================  %
function Y = generate1ofJLabel(originalY,Ntrain,J)
    
    % Generate class label according to 1 of J
    Y = zeros(Ntrain,J);
    for i=1:Ntrain
        column = originalY(i);
        Y(i,column) = 1;
    end
    
end
% ===========================================================  %

% ===========================================================  %
function [C_optimal,D_optimal] = findOptimalHyperparameters(Xtrain,Ytrain,Ntrain,K)
    
    % Split training data into trainVal and testVal
    CVHoldOut = cvpartition(Ntrain,'HoldOut',0.25);

    NtrainVal = CVHoldOut.TrainSize;
    NtestVal = CVHoldOut.TestSize;

    XtrainVal = Xtrain(CVHoldOut.training(),:);
    YtrainVal = Ytrain(CVHoldOut.training(),:);

    XtestVal = Xtrain(CVHoldOut.test(),:);
    YtestVal = Ytrain(CVHoldOut.test(),:);
    
    % Find optimal hyperparameter D
    arrayCost = [];
    C = 10e-7;
    while C <= 10e3
        for D=50:50:1000
            % Generate random w (NxK)
            w = rand(K,D)*2-1;
            % Calculate H=X*w (NxD)
            Htrain = 1./(1+(exp(-(XtrainVal*w))));
            Htest = 1./(1+(exp(-(XtestVal*w))));
            % Calculate Beta=(H'*H + I/C)^-1 * H'*Y
            Beta = (inv((eye(D)/C)+(Htrain'*Htrain)))*(Htrain'*YtrainVal);
            % Generate Y = H*Beta
            Ypredicted = Htest*Beta;
            % Calculate cost L
            % L = ((norm(Beta))^2) + (C*(norm((Htest*Beta)-Ypredicted))^2);
            L = norm(YtestVal - Ypredicted);
            row = [L C D];
            arrayCost = [arrayCost;row]; %#ok
        end
        C = C*10;
    end
    
    [~,indexMin] = min(arrayCost(:,1));
    C_optimal = arrayCost(indexMin,2);
    D_optimal = arrayCost(indexMin,3);
    
end
% ===========================================================  %

% ===========================================================  %
function [Ypredicted,CCR,CCR_ELM] = extremeLearningMachine(Xtrain,Ytrain,Xtest,Ytest,Ntrain,Ntest,K,C,D)
    
    % Apply Extreme Learning Machine Algorithm
    % Generate w (K x D)
    w = rand(K,D)*2-1;
    % Calculate H (N x D)
    Htrain = 1./(1+(exp(-(Xtrain*w))));
    Htest = 1./(1+(exp(-(Xtest*w))));
    % Generate Beta (D x J)
    Beta = (inv((eye(D)/C)+(Htrain'*Htrain)))*(Htrain'*Ytrain);
    % Calculate Y (N x J)
    Ypredicted = Htest*Beta;
    
    % Calculate CCR
    predicts = zeros(size(Ypredicted));

    for i=1:size(Ypredicted,1)
        [~, position] = max(Ypredicted(i,:));
        predicts(i, position) = 1;
    end
    CCR = sum(predicts == Ytest)/Ntest;
    CCR_ELM = mean(CCR);
   
end
% ===========================================================  %

% ===========================================================  %
function showResult(C,D,CCR,CCR_ELM)
    String1 = ['Optimal C hyperparameter: ',num2str(C)];
    String2 = ['Optimal D hyperparameter: ',num2str(D)];
    disp(String1);
    disp(String2);
    disp("CCR per class ELM Algorithm: ");
    disp(CCR);
    String3 = ['CCR ELM Algorithm Regularized: ',num2str(CCR_ELM)];
    disp(String3);
    disp(" ")
end
% ===========================================================  %
