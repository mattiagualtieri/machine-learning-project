% Initial configuration 
clearvars;
clc;
rng(1);

% load data from 'data.csv' file
Data = load('data.csv');

fprintf('\t\t\tCCR\t\t\tMAE\t\t\tMMAE\t\ttau\n');

% Define hyperparameters
global C
C = 1000;
global D
D = 200;
global P;
P = 20;
global G;
G = 10;

% X matrix
X = Data(:,4:end);
% Y = S&P
Y = Data(:,1);
[CCR, MAE, MMAE, tau] = EELM(X, Y);
printResults('S&P    ', CCR, MAE, MMAE, tau);
% Y = Moodys
Y = Data(:,2);
[CCR, MAE, MMAE, tau] = EELM(X, Y);
printResults('Moodys', CCR, MAE, MMAE, tau);
% Y = Fitch
Y = Data(:,3);
[CCR, MAE, MMAE, tau] = EELM(X, Y);
printResults('Fitch', CCR, MAE, MMAE, tau);

% --- functions ---

% Evolutionary Extreme Learning Machine function
function [CCR, MAE, MMAE, tau] = EELM(X, Y)

    global C
    global D
    global P;
    global G;

    [~, K] = size(X);
    J = numel(unique(Y));

    % X scaling and normalization
    Xs = scaleData(X);
    Xn = normalizeData(Xs);

    % split X in Xtest and Xtrain
    % using 2007, 2008 and 2009 data for training
    % using 2010 data for test
    Xtrain = Xn(1:81,:);
    Xtest = Xn(82:end,:);
    [Ntrain, ~] = size(Xtrain);
    [Ntest, ~] = size(Xtest);

    % split Y in Ytesting and Ytraining
    % same here as X
    Ytraining = Y(1:81,:);
    Ytesting = Y(82:end,:);

    Ytrain = zeros(Ntrain, J);
    for i = 1:Ntrain
        column = Ytraining(i);
        Ytrain(i, column) = 1;
    end

    Ytest = zeros(Ntest, J);
    for i = 1:Ntest
        column = Ytesting(i);
        Ytest(i, column) = 1;
    end
    
    %findOptimalHyperparameters(Xtrain, Ytrain);
    
    % split into trainval and testval (25% testval 75% trainval)
    cv = cvpartition(Ntrain, 'HoldOut', 0.25);
    index = cv.test;
    Xtrainval = Xtrain(~index,:);
    Ytrainval = Ytrain(~index,:);
    Xtestval = Xtrain(index,:);
    Ytestval = Ytrain(index,:);
    Ntrainval = cv.TrainSize;
    Ntestval = cv.TestSize;
    
    % Generate initial population
    [arrayW, arrayCost] = initPopulation(Xtrainval, Ytrainval, Xtestval, Ytestval);
    % Sort population
    W = sortW(arrayCost, arrayW);
    
    %Doptimal = findOptimalD(Xtrain, Ytrain);
    
    % Evolutionary process
    pInitCrossover = 3;
    pFinalCrossover = 11;
    pInitMutation = pFinalCrossover + 2;
    newW = [];
    for g = 1:1:G
        % Select Elite 
        newW(:,:,1) = W(:,:,1);
        newW(:,:,2) = W(:,:,2);
        
        % Crossover
        for p = pInitCrossover:2:pFinalCrossover 
            % Set parents
            father = W(:,:,p);
            mother = W(:,:,p + 1);
            % Single Point Crossover
            crossoverPoint = randi([1 K]);
            son1 = [father(:,1:crossoverPoint) mother(:,crossoverPoint+1:end)];
            son2 = [mother(:,1:crossoverPoint) father(:,crossoverPoint+1:end)];
            % Add to Generation
            newW(:,:,p) = son1;
            newW(:,:,p + 1) = son2;    
        end
        
        % Mutation
        for p = pInitMutation:1:P
            % Extract w to mutate
            Wp = W(:,:,p);
            % Mutate
            column = randi([1 K]);
            sigma = std(Wp);
            Wp(:,column) = Wp(:,column) + sigma(:,column).* rand(size(Wp, 1),1);
            % Add to Generation 
            newW(:,:,p) = Wp;
        end
        
        % Fitness
        arrayW = [];
        arrayCost = [];
        for p = 1:1:P
            % Get Wp
            Wp = newW(:,:,p);
            % Calculate H 
            Htrain = 1 ./ (1 + (exp(-(Xtrainval * Wp))));
            Htest = 1 ./ (1+(exp(-(Xtestval * Wp))));
            % Get Beta
            Beta = (inv((eye(D)/C) + (Htrain'*Htrain)))*(Htrain'*Ytrainval);
            % Calculate Y
            Ypredicted = Htest * Beta;
            % Calculate cost L
            L = norm(Ytestval - Ypredicted);
            row = [p L];
            arrayCost = [arrayCost; row];
            arrayW(:,:,p) = Wp;
        end
        W = sortW(arrayCost, arrayW);
    end
    Wfinal = W(:,:,1);

    % Apply Extreme Learning Machine Algorithm

    % Calculate H (N x D)
    Htrain = 1 ./ (1 + (exp(-(Xtrain * Wfinal))));
    Htest = 1 ./ (1 + (exp(-(Xtest * Wfinal))));
    % Beta = [(H'*H + I/C)^-1] * (H' * Y)
    % note: Beta is (D x J)
    Beta = (inv(((Htrain'*Htrain) + eye(D)/C))) * (Htrain'*Ytrain);
    % Calculate Y (N x J)
    Ypredicted = Htest * Beta;

    H = size(Ypredicted, 1);
    predicts = zeros(H, 1);
    for i = 1:size(Ypredicted, 1)
        [~, position] = max(Ypredicted(i,:));
        predicts(i) = position;
    end
    
    % CCR --> correct cassification rate
    CCR = sum(predicts == Ytesting)/H;
    % MAE --> mean absolute error
    MAE = sum(abs(predicts - Ytesting))/H;
    % tau --> the Kendall's tau
    tau = corr(predicts, Ytesting, 'type', 'Kendall');
    % MMAE --> maximum MAE
    MMAE = max(abs(predicts - Ytesting));
    
end

% Init population function
function [arrayW, arrayCost] = initPopulation(Xtrain, Ytrain, Xtest, Ytest)
    
    [~, K] = size(Xtrain);
    
    arrayW = [];
    arrayCost = [];
    
    global P;
    global C;
    global D;
    for p = 1:1:P
        Wp = rand(K, D)*2 - 1;
        % Calculate H = X * W (N x D)
        Htrain = 1 ./ (1 + (exp(-(Xtrain * Wp))));
        Htest = 1 ./ (1 + (exp(-(Xtest * Wp))));
        % Beta = [(H'*H + I/C)^-1] * (H' * Y)
        % note: Beta is (D x J)
        Beta = (inv(((Htrain'*Htrain) + eye(D)/C))) * (Htrain'*Ytrain);
        % Calculate Y = H * Beta
        Ypredicted = Htest * Beta;
        % Calculate cost L
        L = norm(Ytest - Ypredicted);
        row = [p L];
        arrayCost = [arrayCost; row];
        arrayW(:,:,p) = Wp;
    end
    
end

% Sort W function
function [W] = sortW(arrayCost, arrayW)
    arrayCostSorted = sortrows(arrayCost, 2);
    W = [];
    for i = 1:1:size(arrayCostSorted, 1)
        index = arrayCostSorted(i,1);
        wi = arrayW(:,:,index);
        W(:,:,i) = wi;
    end
end

% Find optimal hyperparameters function
function [C, D] = findOptimalHyperparameters(X, Y)

    global C;
    global D;

    [N, K] = size(X);
    % split into test and train (25% val, 75% trainval)
    cv = cvpartition(N, 'HoldOut', 0.25);
    index = cv.test;
    Xtrain = X(~index,:);
    Ytrain = Y(~index,:);
    Xtest = X(index,:);
    Ytest = Y(index,:);
    Ntrain = cv.TrainSize;
    Ntest = cv.TestSize;
    
    C = 10e-3;
    parametersMatrix = zeros(20, 2);
    i = 1;
    while C <= 10e3
        for D = 50:50:1000
            % Get W random matrix (K x D)
            W = rand(K, D)*2 - 1;
            % Bias
            bias_train_vector = rand(Ntrain, 1);
            bias_train = bias_train_vector(:,ones(1, D));
            bias_test_vector = rand(Ntest, 1);
            bias_test = bias_test_vector(:,ones(1, D));
            % Compute H
            Htrain = 1 ./ (1 + (exp(-(Xtrain * W + bias_train))));
            Htest = 1 ./ (1 + (exp(-(Xtest * W + bias_test))));
            % Get Beta matrix (D x J)
            Beta = (inv((eye(D)/C)+(Htrain'*Htrain)))*(Htrain'*Ytrain);
            % Ypredicted = H * Beta
            Ypredicted = Htest * Beta;
            % Parameters matrix
            L = norm(Ytest - Ypredicted);
            % every step we add arrayCost the row [L D]
            parametersMatrix(i, 1) = C;
            parametersMatrix(i, 2) = D;
            parametersMatrix(i, 3) = L;
            i = i + 1;
        end
        C = C * 10;
    end

    % Find optimal C and D
    [~, index] = min(parametersMatrix(:,3));
    C = parametersMatrix(index, 1);
    D = parametersMatrix(index, 2);
end

% Data scaling function
function [Xs] = scaleData(X)
    Xs = (X - min(X))./(max(X)-min(X));
end

% Data normalization function
function [Xn] = normalizeData(X)
    Xn = (X -mean(X))./(std(X));
end

% Print results function
function printResults(name, CCR, MAE, MMAE, tau)
    fprintf('%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n', name, CCR, MAE, MMAE, tau);
end