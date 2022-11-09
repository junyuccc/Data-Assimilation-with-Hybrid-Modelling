function [Wout,rtotal,traindata,trainoutput] = RC_train(Data,Win,Wres,resSize,gamma,initialen,beta)
%RC_train:set up the function of RC training
[~,T] = size(Data);
trainlen = T-initialen;
len = initialen + trainlen;
% training period
r = zeros(resSize,1);
rtotal = zeros(resSize,len);
for t = 1:len
    ut = Data(:,t);
    r = (1 - gamma)*r + gamma*tanh(Win*ut + Wres*r);
    rtotal(:,t) = r;
end
rtrain = rtotal(:,initialen:len-1);
rtrain(resSize/2+1:resSize,:) = rtrain(resSize/2+1:resSize,:).^2;
traindata = Data(:,initialen+1:len);
Wout = ((rtrain*rtrain' + beta*eye(resSize)) \ (rtrain*traindata'))';
trainoutput = Wout*rtrain;
end

