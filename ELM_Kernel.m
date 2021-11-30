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

% ELM Kernel for agency 1
if agency == 1
    [Ypredicted,CCR,C,sigma] = elmKernel(Xtrain,Xtest,Ytrain1,Ytest1,Ntrain,Ntest);
    disp("=======================================================================");
    disp("RATE AGENCY 1) S&P: ")
    showResult(C,sigma,CCR);
    disp("=======================================================================");
end
% ELM Kernel for agency 2
if agency == 2
    [Ypredicted,CCR,C,sigma] = elmKernel(Xtrain,Xtest,Ytrain2,Ytest2,Ntrain,Ntest);
    disp("=======================================================================");
    disp("RATE AGENCY 2) Moodys: ")
    showResult(C,sigma,CCR);
    disp("=======================================================================");
end
% ELM Kernel for agency 3
if agency == 3
    [Ypredicted,CCR,C,sigma] = elmKernel(Xtrain,Xtest,Ytrain3,Ytest3,Ntrain,Ntest);
    disp("=======================================================================");
    disp("RATE AGENCY 3) Fitch: ")
    showResult(C,sigma,CCR);
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
function [Ypredicted,CCR,C,sigma] = elmKernel(Xtrain,Xtest,Ytrain,Ytest,Ntrain,Ntest)
    % Get optimal value of C and sigma
    [C,sigma] = findOptimalHyperparameters(Xtrain,Ytrain,Ntrain);
    % Apply Kernel Extreme Learning Machine
    [Ypredicted,CCR] = kernelExtremeLearningMachine(Xtrain,Ytrain,Xtest,Ytest,Ntrain,Ntest,C,sigma);
end
% ======================================================================= %

% ======================================================================= %
function [C_optimal,sigma_optimal] = findOptimalHyperparameters(Xtrain,Ytrain,Ntrain)

    % Split training data into trainVal and testVal
    CVHoldOut = cvpartition(Ntrain,'HoldOut',0.25);

    NtrainVal = CVHoldOut.TrainSize;
    NtestVal = CVHoldOut.TestSize;

    XtrainVal = Xtrain(CVHoldOut.training(),:);
    YtrainVal = Ytrain(CVHoldOut.training(),:);

    XtestVal = Xtrain(CVHoldOut.test(),:);
    YtestVal = Ytrain(CVHoldOut.test(),:);
    
    % Find optimal hyperparameters C and sigma
    arrayCost = [];
    
    C = 10e-3;
    while C <= 10e3
        sigma = 10e-3;
        while sigma <= 10e3
            % Generate Omega matrix -> omega = H'*H
            omega = generateOmega(XtrainVal,NtrainVal,sigma);
            %omega = generateKernelMatrix(XtrainVal,XtrainVal,NtrainVal,NtrainVal,sigma);
            % Calculate kernel matrix H*H'
            M = generateKernelMatrix(XtrainVal,XtestVal,NtrainVal,NtestVal,sigma);
            % Calculate output Y -> Y = H*H'*(I/C + omega)^-1 * Y
            Ypredicted = M * (inv((eye(size(omega))/C) + omega)) * YtrainVal;
            % Calculate cost L
            L = norm(YtestVal - Ypredicted);
            row = [L C sigma];
            arrayCost = [arrayCost;row]; %#ok
            sigma = sigma * 10;
        end
        C = C * 10;
    end
    
    [~,indexMin] = min(arrayCost(:,1));
    C_optimal = arrayCost(indexMin,2);
    sigma_optimal = arrayCost(indexMin,3);

end
% ======================================================================= %

% ======================================================================= %
function [predicts,CCR] = kernelExtremeLearningMachine(Xtrain,Ytrain,Xtest,Ytest,Ntrain,Ntest,C,sigma)
    
    % Generate omega
    omega = generateOmega(Xtrain,Ntrain,sigma);
    % Calculate kernel matrix H*H'
    M = generateKernelMatrix(Xtrain,Xtest,Ntrain,Ntest,sigma);
    % Calculate output Y 
    Ypredicted = M * (inv((eye(size(omega))/C) + omega)) * Ytrain;
    
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
function omega = generateOmega(X,N,sigma)
    % Calcultate omega -> omega = H'*H
    omega = zeros(N,N);
    for i=1:1:N
        for j=1:1:N
            GK = gaussianKernel(X(i,:),X(j,:),sigma);
            omega(i,j) = GK;
        end
    end
end
% ======================================================================= %

% ======================================================================= %
function KM = generateKernelMatrix(X1,X2,N1,N2,sigma)
    KM = zeros(N2,N1);
    for i=1:1:N2
        for j=1:1:N1
            GK = gaussianKernel(X2(i,:),X1(j,:),sigma);
            KM(i,j) = GK;
        end
    end
end
% ======================================================================= %

% ======================================================================= %
function GK = gaussianKernel(x,y,sigma)
    % Radial Basis Function 
    aux1 = norm(x-y);
    aux2 = sigma * sigma;
    aux3 = -(aux1/aux2);
    GK = exp(aux3);
end
% ======================================================================= %

% ======================================================================= %
function showResult(C,sigma,CCR)
    String1 = ['Optimal C hyperparameter: ',num2str(C)];
    String2 = ['Optimal sigma hyperparameter: ',num2str(sigma)];
    disp(String1);
    disp(String2);
    String3 = ['CCR Kernel ELM Algorithm: ',num2str(CCR)];
    disp(String3);
end
% ======================================================================= %



