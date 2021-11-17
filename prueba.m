
%Cargamos la base de datos
base_de_datos = load('base_de_datos.csv');

%Matriz X
X = base_de_datos(:,4:end);
%Matriz Y
Y = base_de_datos(:,1:3);

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


%Matriz CV con un 25% dedicado a Test y un 75% a Train
cv = cvpartition(Ntrain, 'Holdout', 0.25);
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