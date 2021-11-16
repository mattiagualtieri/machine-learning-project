% ===========================================================  %

% KERNEL KNN ALGORITHM

% Initial configuration 
clear all;
clc;
rng(1);
global sigma;
D = load('spambase.csv');

% Characterization of the dataset 
X = D(:,1:end-1);
Y = D(:,end);
[N, K] = size(X);
J = numel(unique(Y));

%Reduccion de dimensionalidad
%{
HOInicial = cvpartition(N, 'HoldOut', 0.9);
X = D(HOInicial.training,1:end-1);
Y = D(HOInicial.training,end);
[N, K] = size(X);
%}


% Data normalization y SCALED:
Xe = (X - min(X)) ./ (max(X)-min(X));
Xn = (X - mean(X)) ./ std(X);

% Primera particion
particion = cvpartition(N, 'HoldOut', 0.25);


Xtrain = Xe(particion.training(),:);
Xtest = Xe(particion.test(),:);
Ytrain = Y(particion.training(),:);
Ytest = Y(particion.test(),:);

XtrainSize = size(Xtrain,1);

% Segunda particion
particion2 = cvpartition(XtrainSize, 'HoldOut', 1/4);

XtrainVal = Xtrain(particion2.training(),:);
Xval = Xtrain(particion2.test(),:);
YtrainVal = Ytrain(particion2.training(),:);
Yval = Ytrain(particion2.test(),:);

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
        [Neighbours] = knnsearch(XtrainVal, Xval, 'K',K(i),'Distance',@EK);
        LabelsPerPattern = YtrainVal(Neighbours);
        LabelsKNN = mode(LabelsPerPattern')';
        ccrKNN = sum(LabelsKNN == Yval)/particion2.TestSize;
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

[Neighbours] = knnsearch(Xtrain,Xtest,'K',k_optimal, 'Distance', @EK);
LabelsPerPattern = Ytrain(Neighbours);
LabelsKNN = mode(LabelsPerPattern')';
CCR_kNN = sum(LabelsKNN == Ytest)/particion.TestSize

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

