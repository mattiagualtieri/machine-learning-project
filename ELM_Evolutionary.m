% ===========================================================  %

% Extreme Learning Machine Evolutionary Algorithm 
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

% Define value of C and D hyperparameters
C = 1000;
D = 200;

% Define number of generations G and population size P
G = 10;
P = 20;

% Evolutionary Extreme Learning Machine agency 1 
w1 = evolutionaryExtremeLearningMachine(Xtrain,Ytrain1,Ntrain,K,C,D,P,G);
[Ypredicted1,CCR1,CCR_ELM1] = extremeLearningMachine(Xtrain,Ytrain1,Xtest,Ytest1,Ntrain,Ntest,C,D,w1);
disp("RATE AGENCY 1) S&P: ")
showResult(C,D,CCR1,CCR_ELM1);

% Evolutionary Extreme Learning Machine agency 2
w2 = evolutionaryExtremeLearningMachine(Xtrain,Ytrain2,Ntrain,K,C,D,P,G);
[Ypredicted2,CCR2,CCR_ELM2] = extremeLearningMachine(Xtrain,Ytrain2,Xtest,Ytest2,Ntrain,Ntest,C,D,w2);
disp("RATE AGENCY 2) Moodys: ")
showResult(C,D,CCR2,CCR_ELM2);

% Evolutionary Extreme Learning Machine agency 3
w3 = evolutionaryExtremeLearningMachine(Xtrain,Ytrain3,Ntrain,K,C,D,P,G);
[Ypredicted3,CCR3,CCR_ELM3] = extremeLearningMachine(Xtrain,Ytrain3,Xtest,Ytest3,Ntrain,Ntest,C,D,w3);
disp("RATE AGENCY 3) Fitch: ")
showResult(C,D,CCR3,CCR_ELM3);

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
function w = evolutionaryExtremeLearningMachine(Xtrain,Ytrain,Ntrain,K,C,D,P,G)
    
    % Split training data into trainVal and testVal
    CVHoldOut = cvpartition(Ntrain,'HoldOut',0.25);

    NtrainVal = CVHoldOut.TrainSize;
    NtestVal = CVHoldOut.TestSize;

    XtrainVal = Xtrain(CVHoldOut.training(),:);
    YtrainVal = Ytrain(CVHoldOut.training(),:);

    XtestVal = Xtrain(CVHoldOut.test(),:);
    YtestVal = Ytrain(CVHoldOut.test(),:);
    
    % Generate initial population
    [arrayW,arrayCost] = initPopulation(XtrainVal,YtrainVal,XtestVal,YtestVal,K,C,D,P);
    % Sort population
    W = sortW(arrayCost,arrayW);
    
    % Evolutionary process
    pInitCrossover = 3;
    pFinalCrossover = 11;
    pInitMutation = pFinalCrossover + 2;
    newW = [];
    for g=1:1:G
        % Select Elite 
        newW(:,:,1) = W(:,:,1);
        newW(:,:,2) = W(:,:,2);
        
        % Crossover
        for p=pInitCrossover:2:pFinalCrossover 
            % Set parents
            father = W(:,:,p);
            mother = W(:,:,p+1);
            % Single Point Crossover
            crossoverPoint = randi([1 K]);
            son1 = [father(:,1:crossoverPoint) mother(:,crossoverPoint+1:end)];
            son2 = [mother(:,1:crossoverPoint) father(:,crossoverPoint+1:end)];
            % Add to Generation
            newW(:,:,p) = son1;
            newW(:,:,p+1) = son2;    
        end
        
        % Mutation
        for p=pInitMutation:1:P
            
            % Extract w to mutate
            wp = W(:,:,p);
            % Mutate
            column = randi([1 K]);
            sigma = std(wp);
            wp(:,column) = wp(:,column) + sigma(:,column).* rand(size(wp,1),1);
            % Add to Generation 
            newW(:,:,p) = wp;
        end
        
        % Calculate fitness
        arrayW = [];
        arrayCost = [];
        for p=1:1:P
            % Get w
            wp = newW(:,:,p);
            % Calculate H 
            Htrain = 1./(1+(exp(-(XtrainVal*wp))));
            Htest = 1./(1+(exp(-(XtestVal*wp))));
            % Get Beta
            Beta = (inv((eye(D)/C)+(Htrain'*Htrain)))*(Htrain'*YtrainVal);
            % Calculate Y
            Ypredicted = Htest*Beta;
            % Calculate cost L
            L = norm(YtestVal - Ypredicted);
            row = [p L];
            arrayCost = [arrayCost; row]; %#ok
            arrayW(:,:,p) = wp; %#ok
        end
        W = sortW(arrayCost,arrayW);    
    end
    w = W(:,:,1);
end
% ===========================================================  %

% ===========================================================  %
function [arrayW,arrayCost] = initPopulation(XtrainVal,YtrainVal,XtestVal,YtestVal,K,C,D,P)
    
    % Generate first population
    arrayW = [];
    arrayCost = [];
    for p=1:1:P
        % Generate w (N x K)
        wp = rand(K,D)*2-1;
        % Calculate H = X*w (N x D)
        Htrain = 1./(1+(exp(-(XtrainVal*wp))));
        Htest = 1./(1+(exp(-(XtestVal*wp))));
        % Generate Beta = (H'*H + I/C)^-1 * H'*Y 
        Beta = (inv((eye(D)/C)+(Htrain'*Htrain)))*(Htrain'*YtrainVal);
        % Calculate Y = H*Beta
        Ypredicted = Htest*Beta;
        % Calculate cost L
        L = norm(YtestVal - Ypredicted);
        row = [p L];
        arrayCost = [arrayCost; row]; %#ok
        arrayW(:,:,p) = wp; %#ok  
    end
end
% ===========================================================  %

% ===========================================================  %
function W = sortW(arrayCost,arrayW)
    % Sort
    arrayCostSorted = sortrows(arrayCost,2);
    W = [];
    for i=1:1:size(arrayCostSorted,1)
        index = arrayCostSorted(i,1);
        wi = arrayW(:,:,index);
        W(:,:,i) = wi; %#ok 
    end
end
% ===========================================================  %

% ===========================================================  %
function [Ypredicted,CCR,CCR_ELM] = extremeLearningMachine(Xtrain,Ytrain,Xtest,Ytest,Ntrain,Ntest,C,D,w)
    
    % Extreme Learning Machine algorithm
    % Calculate H 
    Htrain = 1./(1+(exp(-(Xtrain*w))));
    Htest = 1./(1+(exp(-(Xtest*w))));
    % Calculte Beta
    Beta = (inv((eye(D)/C)+(Htrain'*Htrain)))*(Htrain'*Ytrain);
    % Calculate Y
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



