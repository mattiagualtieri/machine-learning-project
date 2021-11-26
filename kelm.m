% Initial configuration 
clearvars;
clc;
rng(1);

% load data from 'data.csv' file
D = load('data.csv');

% X matrix
X = D(:,4:end);
% Y = S&P
Y = D(:,1);
fprintf('Starting S&P...\n');
CCR = KELM(X, Y);
fprintf('Terminated Kernel ELM algorithm\n');
string = ['CCR = ', num2str(CCR)];
fprintf(string);
fprintf('\n');
string = ['Mean CCR = ', num2str(mean(CCR))];
fprintf(string);
fprintf('\n\n');


% --- functions ---

% Kernel Extreme Learning Machine function
function [CCR] = KELM(X, Y)

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

    % split Y in Ytest and Ytrain
    % same here as X
    % note: for the training set, we translate the only column of Y into a matrix
    % note: the matrix is (Ntrain x J)
    Ytrain = fromColumnToMatrix(Y(1:81,:));
    Ytest = fromColumnToMatrix(Y(82:end,:));

    [C, sigma] = findOptimalHyperparameters(Xtrain, Ytrain);
    fprintf('Found optimal hyperparameters\n');
    fprintf('C = %d\n', C);
    fprintf('sigma = %d\n', sigma);

    % Apply Kernel Extreme Learning Machine
    % Generate omega
    Omega = generateOmegaMatrix(Xtrain, sigma);
    % Calculate kernel matrix H*H'
    KernelMatrix = generateKernelMatrix(Xtrain, Xtest, sigma);
    % Calculate output Y
    Ypredicted = KernelMatrix * (inv((eye(size(Ntrain))/C) + Omega)) * Ytrain;

    % Calculate CCR
    predicts = zeros(size(Ypredicted));
    for i = 1:size(Ypredicted,1)
        [~, position] = max(Ypredicted(i,:));
        predicts(i, position) = 1;
    end
    CCR = sum(predicts == Ytest)/Ntest;

end

% Find optimal hyperparameters function
function [C, sigma] = findOptimalHyperparameters(X, Y)

    [N, ~] = size(X);
    % split into test and train (25% val, 75% train)
    cv = cvpartition(N, 'HoldOut', 0.25);
    index = cv.test;
    Xtrain = X(~index,:);
    Ytrain = Y(~index,:);
    Xtest = X(index,:);
    Ytest = Y(index,:);
    Ntrain = cv.TrainSize;
    
    parametersMatrix = zeros(49, 3);
    i = 1;
    C = 10e-3;
    while C <= 10e3
        sigma = 10e-3;
        while sigma <= 10e3
            Omega = generateOmegaMatrix(Xtrain, sigma);
            % KernelMatrix = H*H'
            KernelMatrix = generateKernelMatrix(Xtrain, Xtest, sigma);
            % Y = H*H' * (I/C + Omega)^-1 * Y
            Ypredicted = KernelMatrix * (inv((eye(Ntrain)/C) + Omega)) * Ytrain;
            % Calculate cost L
            L = norm(Ytest - Ypredicted);
            % every step we add arrayCost the row [L D]
            parametersMatrix(i, 1) = C;
            parametersMatrix(i, 2) = sigma;
            parametersMatrix(i, 3) = L;
            i = i + 1;
            sigma = sigma * 10;
        end
        C = C * 10;
    end
    
    [~, index] = min(parametersMatrix(:,3));
    C = parametersMatrix(index, 1);
    sigma = parametersMatrix(index, 2);

end

% Generate Kernel Matrix function
function KernelMatrix = generateKernelMatrix(Xtrain, Xtest, sigma)
    [Ntrain, ~] = size(Xtrain);
    [Ntest, ~] = size(Xtest);
    KernelMatrix = zeros(Ntest, Ntrain);
    for i = 1:1:Ntest
        for j = 1:1:Ntrain
            dist = gaussianKernel(Xtest(i,:), Xtrain(j,:), sigma);
            KernelMatrix(i,j) = dist;
        end
    end
end

% Generate Omega function
function [Omega] = generateOmegaMatrix(X, sigma)
    [N, ~] = size(X);
    % Omega = H' * H
    Omega = zeros(N, N);
    for i = 1:1:N
        for j = 1:1:N
            dist = gaussianKernel(X(i,:), X(j,:), sigma);
            Omega(i, j) = dist;
        end
    end
end

% Gaussian Kernel function
function [dist] = gaussianKernel(X, Y, sigma)
    arg = (norm(X - Y)) / (sigma^2);
    dist = exp((-1)*arg);
end

% From column to matrix function
function [Y] = fromColumnToMatrix(Yoriginal)
    [N, ~] = size(Yoriginal);
    J = numel(unique(Yoriginal));
    Y = zeros(N, J);
    for i = 1:N
        column = Yoriginal(i);
        Y(i, column) = 1;
    end
end

% From matrix to column function
function [Y] = fromMatrixToColumn(Yoriginal)
    H = size(Yoriginal, 1);
    Y = zeros(H, 1);
    for i = 1:H
        [~, position] = max(Yoriginal(i,:));
        Y(i) = position;
    end
end

% Data scaling function
function [Xs] = scaleData(X)
    Xs = (X - min(X))./(max(X)-min(X));
end

% Data normalization function
function [Xn] = normalizeData(X)
    Xn = (X - min(X)) ./ (max(X) - min(X));
end