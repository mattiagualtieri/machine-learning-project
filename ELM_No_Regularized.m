clear all;

rng(1);

%Cargamos la base de datos
base_de_datos = load('base_de_datos.csv');

%Matriz X
X = base_de_datos(:,4:end);
%Matriz Y
Y = base_de_datos(:,1:3);

J = numel(unique(Y));
[N,K] = size(X);

%Escalado de la matriz X
Xscaled = (X - min(X))./(max(X)-min(X));
%Normalizacion de la matriz Y
Xnormalized = (X -mean(X))./(std(X));

Xtrain = X(1:81,:);
YtrainSP = Y(1:81,1);
YtrainMoody = Y(1:81,2);
YtrainFitch = Y(1:81,3);
Ntrain = size(Xtrain,1);

Xtest = X(82:end,:);
YtestSP = Y(82:end,1);
YtestMoody = Y(82:end,2);
YtestFitch = Y(82:end,3);
Ntest = size(Xtest,1);

YtrainSP = generate1ofJLabel(YtrainSP,Ntrain,J);
YtrainMoody = generate1ofJLabel(YtrainMoody,Ntrain,J);
YtrainFitch = generate1ofJLabel(YtrainFitch,Ntrain,J);

YtestSP = generate1ofJLabel(YtestSP,Ntest,J);
YtestMoody = generate1ofJLabel(YtestMoody,Ntest,J);
YtestFitch = generate1ofJLabel(YtestFitch,Ntest,J);

% Dividimos la matriz de training en Test y Validating para tener unos
% datos con los que validar nuestro entrenamiento del algoritmo. 
cv = cvpartition(Ntrain,'HoldOut',0.25);

%Matriz de índices (0 = Train, 1 = Test)
index = cv.test;
%Matriz resultante de X para Training
Xtrainval = Xtrain(~index,:);
Xval = Xtrain(index,:);
%Matriz resultante de Y para Training de la agencia SP
YtrainvalSP = YtrainSP(~index,:);
YvalSP = YtrainSP(index,:);
%Matriz resultante de Y para Training de la agencia Moody
YtrainvalMoody = YtrainMoody(~index,:);
YvalMoody = YtrainMoody(index,:);
%Matriz resultante de Y para Training de la agencia Fitch
YtrainvalFitch = YtrainFitch(~index,:);
YvalFitch = YtrainFitch(index,:);


%-------------------------------------------------------------------------------------------------------------

%AGENCIA SP

%Encontramos el hiperparametro optimo D para SP
i = 1;
delta = 10e-3;
for D=50:50:2000
    w = rand(K,D)*2-1;
    Htrain = 1./(1+(exp(-(Xtrainval*w))));
    Htest = 1./(1+(exp(-(Xval*w))));
    Beta = (inv((Htrain'*Htrain) +(delta*eye(size(Htrain,2)))))*Htrain'*YtrainvalSP;
    Ypredicted = Htest*Beta;
    L = norm(YvalSP - Ypredicted);
    rowSP(i,1) = L;
    rowSP(i,2) = D;
    i = i + 1;
end
L_optimal_SP = min(rowSP(:,1));
[L_row, L_col] = find(rowSP == L_optimal_SP);
D_optimal_SP = rowSP(L_row,2);
%D_optimal_SP = min(row(:,2));

%Ahora ejecutamos el algoritmo ELM con el hiperparametro optimo de SP
w = rand(K,D_optimal_SP)*2-1;
Htrain = 1./(1+(exp(-(Xtrain*w))));
Htest = 1./(1+(exp(-(Xtest*w))));
Beta = (inv((Htrain'*Htrain) +(delta*eye(size(Htrain,2)))))*Htrain'*YtrainSP;
Ypredicted = Htest*Beta;
    
% Calculamos el CCR de la Agencia SP
predicts = zeros(size(Ypredicted));
for i=1:size(Ypredicted,1)
    [~, position] = max(Ypredicted(i,:));
    predicts(i, position) = 1;
end
CCR_SP = sum(predicts == YtestSP)/Ntest;
CCR_ELM_SP = mean(CCR_SP);



%-------------------------------------------------------------------------------------------------------------

%AGENCIA MOODY

%Encontramos el hiperparametro optimo D para Moody
i = 1;
for D=50:50:2000
    w = rand(K,D)*2-1;
    Htrain = 1./(1+(exp(-(Xtrainval*w))));
    Htest = 1./(1+(exp(-(Xval*w))));
    Beta = (inv((Htrain'*Htrain) +(delta*eye(size(Htrain,2)))))*Htrain'*YtrainvalMoody;
    Ypredicted = Htest*Beta;
    L = norm(YvalMoody - Ypredicted);
    rowMoody(i,1) = L;
    rowMoody(i,2) = D;
    i = i + 1;
end
L_optimal_Moody = min(rowMoody(:,1));
[L_row, L_col] = find(rowMoody == L_optimal_Moody);
D_optimal_Moody = rowMoody(L_row,2);
%D_optimal_Moody = min(row(:,2));

%Ahora ejecutamos el algoritmo ELM con el hiperparametro optimo de Moody
w = rand(K,D_optimal_Moody)*2-1;
Htrain = 1./(1+(exp(-(Xtrain*w))));
Htest = 1./(1+(exp(-(Xtest*w))));
Beta = (inv((Htrain'*Htrain) +(delta*eye(size(Htrain,2)))))*Htrain'*YtrainMoody;
Ypredicted = Htest*Beta;
    
% Calculamos el CCR de la Agencia Moody
predicts = zeros(size(Ypredicted));
for i=1:size(Ypredicted,1)
    [~, position] = max(Ypredicted(i,:));
    predicts(i, position) = 1;
end
CCR_Moody = sum(predicts == YtestMoody)/Ntest;
CCR_ELM_Moody = mean(CCR_Moody);



%-------------------------------------------------------------------------------------------------------------



%AGENCIA FITCH

%Encontramos el hiperparametro optimo D para Fitch
i = 1;
for D=50:50:2000
    w = rand(K,D)*2-1;
    Htrain = 1./(1+(exp(-(Xtrainval*w))));
    Htest = 1./(1+(exp(-(Xval*w))));
    Beta = (inv((Htrain'*Htrain) +(delta*eye(size(Htrain,2)))))*Htrain'*YtrainvalFitch;
    Ypredicted = Htest*Beta;
    L = norm(YvalFitch - Ypredicted);
    rowFitch(i,1) = L;
    rowFitch(i,2) = D;
    i = i + 1;
end
L_optimal_Fitch = min(rowFitch(:,1));
[L_row, L_col] = find(rowFitch == L_optimal_Fitch);
D_optimal_Fitch = rowFitch(L_row,2);
%D_optimal_Fitch = min(row(:,2));

%Ahora ejecutamos el algoritmo ELM con el hiperparametro optimo de Fitch
w = rand(K,D_optimal_Fitch)*2-1;
Htrain = 1./(1+(exp(-(Xtrain*w))));
Htest = 1./(1+(exp(-(Xtest*w))));
Beta = (inv((Htrain'*Htrain) +(delta*eye(size(Htrain,2)))))*Htrain'*YtrainFitch;
Ypredicted = Htest*Beta;
    
% Calculamos el CCR de la Agencia Fitch
predicts = zeros(size(Ypredicted));
for i=1:size(Ypredicted,1)
    [~, position] = max(Ypredicted(i,:));
    predicts(i, position) = 1;
end
CCR_Fitch = sum(predicts == YtestFitch)/Ntest;
CCR_ELM_Fitch = mean(CCR_Fitch);



%-------------------------------------------------------------------------------------------------------------



function Y = generate1ofJLabel(originalY,Ntrain,J)
    
    % Generate class label according to 1 of J
    Y = zeros(Ntrain,J);
    for i=1:Ntrain
        column = originalY(i);
        Y(i,column) = 1;
    end
    
end
