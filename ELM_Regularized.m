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

% 1 of J 
Ytrain1 = generate1ofJLabel(Ytrain1,Ntrain,J);
Ytrain2 = generate1ofJLabel(Ytrain2,Ntrain,J);
Ytrain3 = generate1ofJLabel(Ytrain3,Ntrain,J);

Ytest1 = generate1ofJLabel(Ytest1,Ntest,J);
Ytest2 = generate1ofJLabel(Ytest2,Ntest,J);
Ytest3 = generate1ofJLabel(Ytest3,Ntest,J);

% Select Rating Agency 
agency = 3;  % 1 = S&P //  2 = Moodys // 3 = Fitch

% ELM Regularized for agency 1
if agency == 1
    [Ypredicted,CCR,C,D] = elmRegularized(Xtrain,Xtest,Ytrain1,Ytest1,Ntrain,Ntest,K);
    disp("=======================================================================");
    disp("RATE AGENCY 1) S&P: ")
    showResult(C,D,CCR);
    disp("=======================================================================");
end
% ELM Regularized for agency 2
if agency == 2
    [Ypredicted,CCR,C,D] = elmRegularized(Xtrain,Xtest,Ytrain2,Ytest2,Ntrain,Ntest,K);
    disp("=======================================================================");
    disp("RATE AGENCY 2) Moodys: ")
    showResult(C,D,CCR);
    disp("=======================================================================");
end
% ELM Regularized for agency 3
if agency == 3
    [Ypredicted,CCR,C,D] = elmRegularized(Xtrain,Xtest,Ytrain3,Ytest3,Ntrain,Ntest,K);
    disp("=======================================================================");
    disp("RATE AGENCY 3) Fitch: ")
    showResult(C,D,CCR);
    disp("=======================================================================");
end
% Error
if agency < 1 || agency > 3
    disp("Error selecting agency");
end
% ======================================================================= %

% ======================================================================= %
function Y = generate1ofJLabel(originalY,Ntrain,J)
    
    % Generate class label according to 1 of J
    Y = zeros(Ntrain,J);
    for i=1:Ntrain
        column = originalY(i);
        Y(i,column) = 1;
    end
    
end
% ======================================================================= %

% ======================================================================= %
function [Ypredicted,CCR,C,D] = elmRegularized(Xtrain,Xtest,Ytrain,Ytest,Ntrain,Ntest,K)
    % Get optimal value of D
    [C,D] = findOptimalHyperparameters(Xtrain,Ytrain,Ntrain,K);
    % Apply Extreme Learning Machine
    [Ypredicted,CCR] = extremeLearningMachine(Xtrain,Ytrain,Xtest,Ytest,Ntest,K,C,D);
end
% ======================================================================= %

% ======================================================================= %
function [C_optimal,D_optimal] = findOptimalHyperparameters(Xtrain,Ytrain,Ntrain,K)
    
    % Split training data into trainVal and testVal
    CVHoldOut = cvpartition(Ntrain,'HoldOut',0.25,'Stratify',false);

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
% ======================================================================= %

% ======================================================================= %
function [predicts,CCR] = extremeLearningMachine(Xtrain,Ytrain,Xtest,Ytest,Ntest,K,C,D)
    
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
    
    newYpredicted = generateNON1ofJLabel(predicts);
    newYtest = generateNON1ofJLabel(Ytest);

    CCR= sum(newYpredicted == newYtest)/Ntest;
   
end
% ======================================================================= %

% ======================================================================= %
function newY = generateNON1ofJLabel(Y)
    [N,J] = size(Y);
    newY = zeros(N,1);
    for n=1:1:N
        for j=1:1:J
            if Y(n,j) == 1
                newY(n) = j;
            end
        end
    end
end
% ======================================================================= %

% ======================================================================= %
function showResult(C,D,CCR)
    String1 = ['Optimal C hyperparameter: ',num2str(C)];
    String2 = ['Optimal D hyperparameter: ',num2str(D)];
    disp(String1);
    disp(String2);
    String3 = ['CCR ELM Algorithm Regularized: ',num2str(CCR)];
    disp(String3);
end
% ======================================================================= %
