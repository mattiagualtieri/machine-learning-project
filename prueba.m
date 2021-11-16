
%Cargamos la base de datos
base_de_datos = load('base_de_datos.csv');

%Matriz X
X = base_de_datos(:,4:end);
%Matriz Y
Y = base_de_datos(:,1);

%Escalado de la matriz X
Xscaled = (X - min(X))./(max(X)-min(X));
%Normalizacion de la matriz Y
Xnormalized = (X -mean(X))./(std(X));

%Matriz CV con un 25% dedicado a Test y un 75% a Train
cv = cvpartition(size(Xscaled, 1), 'Holdout', 0.25);
%Matriz de índices (0 = Train, 1 = Test)
index = cv.test;
%Matriz resultante de X para Training
Xtrainval = Xscaled(~index,:);
Xval = Xscaled(index,:);
%Matriz resultante de Y para Training
Ytrainval = Y(~index,:);
Yval = Y(index,:);