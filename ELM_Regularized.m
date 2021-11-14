% ===========================================================  %

% Extreme Learning Machine Regularized Algorithm 
% Carlos Cuevas Baliñas
% Machine Learning - 4º IITV 

% Initial configuration 
clear all;
rng(1);

% Load dataset 
[Xtraining_scaled, YtrainingOriginal, Xtest_scaled, YtestOriginal, N_training, N_test, K, J] = extractData();

% 1 of J 
Ytraining = zeros(N_training,J);
for i=1:N_training
    column = YtrainingOriginal(i);
    Ytraining(i,column) = 1;
end

Ytest = zeros(N_test,J);
for i=1:N_test
    column = YtestOriginal(i);
    Ytest(i,column) = 1;
end

% Split training data into trainVal and testVal
CVHoldOut = cvpartition(N_training,'HoldOut',0.25);

N_trainVal = CVHoldOut.TrainSize;
N_testVal = CVHoldOut.TestSize;

X_trainVal = Xtraining_scaled(CVHoldOut.training(),:);
Y_trainVal = Ytraining(CVHoldOut.training(),:);

X_testVal = Xtraining_scaled(CVHoldOut.test(),:);
Y_testVal = Ytraining(CVHoldOut.test(),:);

% Find optimal hyperparameters C and D
C = 10e-3;
parametersMatrix = [];

while C <= 10e3
    for D=50:50:1000
        % Get w random matrix (K x D)
        w = rand(K,D)*2-1;
        % Bias
        bias_train_vector = rand(size(X_trainVal,1),1);
        bias_train = bias_train_vector(:,ones(1,D));
        bias_test_vector = rand(size(X_testVal,1),1);
        bias_test = bias_test_vector(:,ones(1,D));
        % Get H 
        H_train = 1./(1+(exp(-(X_trainVal*w+bias_train))));
        H_test = 1./(1+(exp(-(X_testVal*w+bias_test))));
        % Get Beta matrix (D x J)
        Beta = (inv((eye(D)/C)+(H_train'*H_train)))*(H_train'*Y_trainVal);
        % Y = H * Beta
        Y_predicted = H_test * Beta;
        % Parameters matrix
        L = ((norm(Beta))^2) + (C*(norm((H_test*Beta)-Y_predicted))^2);
        row = [C D L];
        parametersMatrix = [parametersMatrix; row]; %#ok
    end
    C = C*10;
end

[~, index] = min(parametersMatrix(:,3));
C_optimal = parametersMatrix(index,1);
D_optimal = parametersMatrix(index,2);

% Apply Extreme Learning Machine Algorithm

w = rand(K,D_optimal)*2-1;

bias_train_vector = rand(size(Xtraining_scaled,1),1);
bias_train = bias_train_vector(:,ones(1,D));
bias_test_vector = rand(size(Xtest_scaled,1),1);
bias_test = bias_test_vector(:,ones(1,D));

H_train = 1./(1+(exp(-(Xtraining_scaled*w+bias_train))));
H_test = 1./(1+(exp(-(Xtest_scaled*w+bias_test))));

Beta = (inv((eye(D)/C)+(H_train'*H_train)))*(H_train'*Ytraining);

Y_predicted = H_test * Beta;

% Calculate CCR
predicts = zeros(size(Y_predicted));

for i=1:size(Y_predicted,1)
    [~, position] = max(Y_predicted(i,:));
    predicts(i, position) = 1;
end

CCR = sum(predicts == Ytest)/N_test;
CCR_ELM = mean(CCR);

% Show results
disp("Optimal C: ");
disp(C_optimal);
disp("Optimal D: ");
disp(D_optimal);
disp("CCR Regularized Extreme Learning Machine:");
disp(CCR);
disp(CCR_ELM);

% ===========================================================  %

function [Xtraining_scaled, Ytraining, Xtest_scaled, Ytest, N_training, N_test, K, J] = extractData()

    % Load database
    DataTraining = load('training.mat');
    TrainingData = DataTraining.TurkiyeEvaluationFilteredWEKAClusteringTraining;
    DataTest= load('testing.mat');
    TestData = DataTest.TurkiyeEvaluationFilteredWEKAClusteringTest;

    % Extract the data
    Xtraining = TrainingData(:,1:end-1);
    Ytraining = TrainingData(:,end);
    Xtest = TestData(:,1:end-1);
    Ytest = TestData(:,end);
    Xtraining_scaled = (Xtraining - min(Xtraining)) ./ (max(Xtraining)-min(Xtraining));
    Xtest_scaled = (Xtest - min(Xtest)) ./ (max(Xtest)-min(Xtest));
    [N_training, K] = size(Xtraining_scaled);
    [N_test, K] = size(Xtest_scaled);
    J = numel(unique(Ytraining));
end