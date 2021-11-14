% ===========================================================  %

% KNN Multiclass Algorithm 
% Carlos Cuevas Baliñas
% Machine Learning - 4º IITV 

% Initial configuration 
clear all;
rng(1);

% Load dataset 
[Xtraining_scaled, Ytraining, Xtest_scaled, Ytest, N_training, N_test, K, J] = extractData();

% Split training data into trainValidation and testValidation
CVHoldOut = cvpartition(N_training,'HoldOut',0.25,'Stratify',false);

N_trainVal = CVHoldOut.TrainSize;
N_testVal = CVHoldOut.TestSize;

X_trainVal = Xtraining_scaled(CVHoldOut.training(),:);
Y_trainVal = Ytraining(CVHoldOut.training(),:);

X_testVal = Xtraining_scaled(CVHoldOut.test(),:);
Y_testVal = Ytraining(CVHoldOut.test(),:);

% Find optimal value of k neighbours
CCR_Array = [];

for k=1:2:13
    [index] = knnsearch(X_trainVal,X_testVal,'K',k,'Distance','euclidean');
    LabelsKNN = Y_trainVal(index);
    Y_predicted = mode(LabelsKNN')';
    k_ccr = sum(Y_predicted == Y_testVal)/N_testVal;
    CCR_Array = [CCR_Array; k_ccr]; %#ok
end

[~, indexMax] = max(CCR_Array'); %#ok
k_optimal = (indexMax * 2) - 1;

% Apply Knn Algorithm with optimal value of k

[index] = knnsearch(Xtraining_scaled,Xtest_scaled,'K',k,'Distance','euclidean');
LabelsKNN = Ytraining(index);
Y_predicted = mode(LabelsKNN')';
CCR = sum(Y_predicted == Ytest)/N_test;

% Show results
disp("Optimal k: ");
disp(k_optimal);
disp("CCR KNN Algorithm:");
disp(CCR);



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