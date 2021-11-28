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

% Define number of generations G and population size P
G = 10;
P = 20;

% Define value of C and D hyperparameters
C = 1000;
D = 200;

% Select Rating Agency 
agency = 2;  % 1 = S&P //  2 = Moodys // 3 = Fitch

% ELM Evolutionary for agency 1
if agency == 1
    [Ypredicted,CCR,CCR_ELM,w] = elmEvolutionary(Xtrain,Xtest,Ytrain1,Ytest1,Ntrain,Ntest,K,C,D,P,G);
    disp("=======================================================================");
    disp("RATE AGENCY 1) S&P: ")
    showResult(C,D,CCR,CCR_ELM);
    disp("=======================================================================");
end
% ELM Evolutionary for agency 2
if agency == 2
    [Ypredicted,CCR,CCR_ELM,w] = elmEvolutionary(Xtrain,Xtest,Ytrain2,Ytest2,Ntrain,Ntest,K,C,D,P,G);
    disp("=======================================================================");
    disp("RATE AGENCY 2) Moodys: ")
    showResult(C,D,CCR,CCR_ELM);
    disp("=======================================================================");
end
% ELM Evolutionary for agency 3
if agency == 3
    [Ypredicted,CCR,CCR_ELM,w] = elmEvolutionary(Xtrain,Xtest,Ytrain3,Ytest3,Ntrain,Ntest,K,C,D,P,G);
    disp("=======================================================================");
    disp("RATE AGENCY 3) Fitch: ")
    showResult(C,D,CCR,CCR_ELM);
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
function [Ypredicted,CCR,CCR_ELM,w] = elmEvolutionary(Xtrain,Xtest,Ytrain,Ytest,Ntrain,Ntest,K,C,D,P,G)
    % Evolutionary Extreme Learning Machine
    w = evolutionaryExtremeLearningMachine(Xtrain,Ytrain,Ntrain,K,C,D,P,G);
    [Ypredicted,CCR,CCR_ELM] = extremeLearningMachine(Xtrain,Ytrain,Xtest,Ytest,Ntest,C,D,w);
end
% ======================================================================= %

% ======================================================================= %
function w = evolutionaryExtremeLearningMachine(Xtrain,Ytrain,Ntrain,K,C,D,P,G)
    
    % Split training data into trainVal and testVal
    CVHoldOut = cvpartition(Ntrain,'HoldOut',0.25);
    
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
% ======================================================================= %

% ======================================================================= %
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
% ======================================================================= %

% ======================================================================= %
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
% ======================================================================= %

% ======================================================================= %
function [predicts,CCR,CCR_ELM] = extremeLearningMachine(Xtrain,Ytrain,Xtest,Ytest,Ntest,C,D,w)
    
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
% ======================================================================= %

% ======================================================================= %
function showResult(C,D,CCR,CCR_ELM)
    String1 = ['Optimal C hyperparameter: ',num2str(C)];
    String2 = ['Optimal D hyperparameter: ',num2str(D)];
    disp(String1);
    disp(String2);
    disp("CCR per class ELM Algorithm: ");
    disp(CCR);
    String3 = ['CCR ELM Algorithm Regularized: ',num2str(CCR_ELM)];
    disp(String3);
end
% ======================================================================= %



