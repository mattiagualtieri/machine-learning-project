% ===========================================================  %
function [Xtrain,Ytrain1,Ytrain2,Ytrain3,Ntrain,Xtest,Ytest1,Ytest2,Ytest3,Ntest,N,J,K] = initData(dataset)

    dataX = dataset(:,4:end);
    dataY = dataset(:,1:3);

    J = numel(unique(dataY));
    [N,K] = size(dataX);

    % Data normalization
    dataX = (dataX - min(dataX))./(max(dataX) - min(dataX));

    % Split data into Train and Test
    Xtrain = dataX(1:81,:);
    Ytrain1 = dataY(1:81,1);
    Ytrain2 = dataY(1:81,2);
    Ytrain3 = dataY(1:81,3);
    Ntrain = size(Xtrain,1);

    Xtest = dataX(82:end,:);
    Ytest1 = dataY(82:end,1);
    Ytest2 = dataY(82:end,2);
    Ytest3 = dataY(82:end,3);
    Ntest = size(Xtest,1);

end
% ===========================================================  %