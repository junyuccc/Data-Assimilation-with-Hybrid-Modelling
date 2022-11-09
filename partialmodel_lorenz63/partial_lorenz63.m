%%% Partial Model filtering for Lorenz63.
%%% Assume we known model equation of the state-x of Lorenz63 without its
%%% observation, while state y&z have noisy observation without model equation. 
%%% Hybrid modeling

clear;clc;close all
%% load the Lorenz truth and noisy data
load('lorenz63_0_16')% truth state;observe data
% estimate observenoise
obs = obs(:,1:2:10000);
truth = truth(:,1:2:10000);
cycle = 1;%iteration numbers
% estimate the observe noise variance
average = mean(obs,2);
obsrmse  = sqrt(mean((obs-average).^2,2));
obsnoise = mean(obsrmse);

%% generate embedding data
timeseries = obs(2:3,:);
m = size(obs,1);
n = size(timeseries,1);% number of unknown-model states
m = m-n;% number of known-model states
delays = 3; % set up delays'number
tau = 2; % set up delay step
delaytime = Embedding(timeseries,delays,tau);% generate delays coordinate
delaytime = [zeros(n*delays,(delays-1)*tau),delaytime];
Data = delaytime(:,:);% get the delay-data for RC training
trainlen=size(Data,2);
%% set up the parameters of unscented kalman filtering 
% and  noise convariance updating 
[N,T] = size(delaytime);
N = N+m; %% sum up the dimension of states being not observed
P = eye(N);
c = sqrt(N-1);
Q1 = randn(N,N)*eye(N)*0.6;
R1 = obsnoise;
Q = Q1;
R = R1;

start = 1;
dt = 0.02; initialen = 20/dt;
xa = [rand(1);delaytime(:,start+1)];
prev_innov  = zeros(2,1);% only n-dimensions states can be observed
prev_FPF = eye(N); 
prev_K = zeros(N,2);
filterstate = zeros(N,T);
% filterstate(:,initialen) = xa;
forex = zeros(N,T);
% forex(:,initialen) = xa;
forecast_x = zeros(N,2*N+1);

%% set up or tune the cyperparameters of RC training
[nn,l] = size(Data);
inSize = nn; resSize = 100; outSize = nn;
rhow_r = 0.67; sigma = 0.44; rhow_in = 0.37; 
gamma = 0.44; beta = 1e-5;
d = 0.05; k = round(resSize*d);

% generate input connection matrix
% Win1 = unifrnd(-0.5,0.5,[resSize inSize]);
Win1 = normrnd(0,rhow_in^2,[resSize inSize]);
adj1 = rand(resSize,inSize);
adj1(adj1 < sigma)=1;
adj1(adj1 ~= 1)=0;
Win = adj1.*Win1;
% generate and reservoir internal connection matrix
adj1 = zeros(resSize,resSize);
for i = 1:resSize
    num = randperm(resSize,k);
    adj1(i,num) = 1;
end
% Wres1 = normrnd(0,1,[resSize resSize]); %生成正态分布随机数
Wres1 = unifrnd(-1,1,[resSize resSize]);
Wres1 = adj1.*Wres1 ;
SR = max(abs(eig(Wres1))) ;
Wres = Wres1 .* ( rhow_r/SR);

%% Iteration of RC training and UKF
for cc=1:cycle
% RC training
[Wout,rtotal,traindata,trainoutput] = RC_train(Data,Win,Wres,resSize,gamma,initialen,beta);
% figure()
% plot(0:dt:10,truth(1,trainlen-20/dt:trainlen-10/dt),'k','linewidth',1);hold on
% plot(0:dt:10,trainoutput(1,end-20/dt:end-10/dt),'r','linewidth',1);hold on;
% plot(0:dt:10,traindata(1,end-20/dt:end-10/dt),'b.','MarkerSize',5);
% title('RC traing');xlabel('t');ylabel('y');
% legend('truth','forecast ','obs')

% rt = repmat(rtotal(:,end),1,7);
rt = rtotal(:,end);
% Rt(:,1) = rt;

%% Unscented kalman filtering
for t =start:T-1

    [U,S,~] = svd(Q);
    Q = U*S*U';
    [U,S,~] = svd(R);
    R = U*S*U';  
    %%% generate unscented ensemble of state xa
    [x_unscented,weights] = unscented(xa,P,c,N);
    weightsmat = diag(weights(2:end));
    
    %%% ensemble kalman filter without a model
%     rt = rtotal(:,t:-tau:t-(delays-1)*tau);
%     rt = rtotal(:,t); 
    %%% unscented kalman filter with Hybrid modeling
    %%% update the state of unknown system equation with RC prediction process
    %%% update the state of known system equation with eqution.
        foredata = x_unscented(2:end,:);
        [fx,rt] = RC_forecast(rt,foredata,Win,Wres,Wout,resSize,gamma);
        forecast_x(2:end,:) = fx(:,:);
        forecast_x(1,:) = x_unscented(1,:) + dt*10*(x_unscented(2,:)-x_unscented(1,:));
%     RR1(:,t+1)=rt;
%     RR2(:,t+1)=rr1;  
%     forecast_x(2,:) = x_unscented(2,:) + dt*( x_unscented(1,:).*(28 - x_unscented(3,:)) - x_unscented(2,:));%差分方程
%     forecast_x(3,:) = x_unscented(3,:) + dt*( x_unscented(1,:).*x_unscented(2,:) - 8/3.*x_unscented(3,:));
    
    mean_x = forecast_x*weights';
    forex(:,t+1) = mean_x;
    Ex = x_unscented(:,2:end) - repmat(xa,1,2*N);
    Exx = forecast_x(:,2:end) - repmat(mean_x,1,2*N);
    FPF = Exx*weightsmat*(Exx');
    Pxx = FPF + Q;
    y = forecast_x(2:3,:);
    mean_y = y*weights';
    Ey = y(:,2:end)-repmat(mean_y,1,2*N);
    HF = Ey/Ex;
    %%% generate updated ensemble then observe it
    [x_unscented,weights]=unscented(mean_x,Pxx,c,N);
    y = x_unscented(2:3,:);
    mean_y = y*weights';
    yy = mean_y(:,:);
    Ex = x_unscented(:,2:end) - repmat(mean_x,1,2*N);
    Ey = y(:,2:end) - repmat(mean_y,1,2*N);
    Pb = Ey*weightsmat*Ey';
    Pyy = Pb + R;
    Pxy = Ex*weightsmat*Ey';%Pxy = Ex*weights*Ey;
    K = Pxy/Pyy;% get the gain of kalman filer
    HT = Pxx\Pxy;
    real_y = obs(2:3,t+1);
    innov = real_y - yy;
    xa = mean_x + K*innov;
    P = Pxx - K*Pyy*K';
    filterstate(:,t+1) = xa;
    %%% the update of model error covariance and observe error covariance
    Gamma0 = prev_innov*prev_innov';
    Gamma1 = innov*prev_innov';
    Rest = Gamma0 - Pb;
    M = (HF\Gamma1 + prev_K*Gamma0)/HT;
    Qest = M - prev_FPF;
    sta = 2000;
    if t>start+20 
        Q=(Q*(2*sta-1)+Qest)/(2*sta);
        R=(R*(sta-1)+Rest)/sta;
    end
    prev_innov = innov;
    prev_K = K;
    prev_FPF = FPF;
    if t>=(500+start)
        RMSEfilter(:,t-(500+start)+1)=sqrt(mean((filterstate(1:m+n,t-500+1:t)-truth(:,t-500+1:t)).^2,2));
        RMSEforecast(:,t-(500+start)+1)=sqrt(mean((forex(1:m+n,t-500+1:t)-truth(:,t-500+1:t)).^2,2));
    end
    
end
%% review results
RMSEFILTER = sqrt(mean((filterstate(1:m+n,start+1:end)-truth(:,start+1:end)).^2,2));
RMSEFORE = sqrt(mean((forex(1:m+n,start+1:end)-truth(:,start+1:end)).^2,2));
RMSEobs = sqrt(mean((obs(:,:)-truth(:,:)).^2,2));
RMSE(cc,:) = mean(sqrt(mean((filterstate(2:m+n,end-80/dt:end)-truth(2:3,end-80/dt:end)).^2,2)));

figure()
subplot(3,1,1);
plot(4501*dt:dt:5000*dt,truth(1,end-10/dt+1:end),'k','linewidth',1);
hold on;
plot(4501*dt:dt:5000*dt,filterstate(1,end-10/dt+1:end),'r','linewidth',1);
hold on;
% plot(obs(1,end-10/dt+1:end),'.','MarkerSize',5);
title('Lorenz63');ylabel('x');
subplot(3,1,2);
plot(4501*dt:dt:5000*dt,truth(2,end-10/dt+1:end),'k','linewidth',1);
hold on;
plot(4501*dt:dt:5000*dt,filterstate(2,end-10/dt+1:end),'r','linewidth',1);
hold on;
plot(4501*dt:dt:5000*dt,obs(2,end-10/dt+1:end),'b.','MarkerSize',5);
legend('truth state','filter state','observe data')
ylabel('y');
subplot(3,1,3);
plot(4501*dt:dt:5000*dt,truth(3,end-10/dt+1:end),'k','linewidth',1);
hold on;
plot(4501*dt:dt:5000*dt,filterstate(3,end-10/dt+1:end),'r','linewidth',1);
hold on;
plot(4501*dt:dt:5000*dt,obs(3,end-10/dt+1:end),'b.','MarkerSize',5);
% plot(forex(1,end-10/dt+1:end),'b.');
xlabel('Time');ylabel('z');

figure()
subplot(3,1,1);
plot(502*dt:dt:5000*dt,RMSEfilter(1,:),'r','linewidth',2);
hold on;
% plot(RMSEforecast(1,:),'b','linewidth',2);
% hold on;
% plot(repmat(RMSEobs(1),1,size(RMSEfilter,2)),'k','linewidth',2)
% hold on;
% plot(repmat(RMSEFILTER(1),1,size(RMSEfilter,2)),'r--','linewidth',2);
% hold on;
% plot(repmat(RMSEFORE(1),1,size(RMSEfilter,2)),'b--','linewidth',2);
ylabel('x');title('Lorenz63');axis([-inf inf,-inf 3]);

subplot(3,1,2);
plot(502*dt:dt:5000*dt,RMSEfilter(2,:),'r','linewidth',2);
hold on;
% plot(RMSEforecast(2,:),'b','linewidth',2);
% hold on;
plot(502*dt:dt:5000*dt,repmat(RMSEobs(2),1,size(RMSEfilter,2)),'k--','linewidth',2)
hold on;
% plot(repmat(RMSEFILTER(2),1,size(RMSEfilter,2)),'r--','linewidth',2)
% hold on;
% plot(repmat(RMSEFORE(2),1,size(RMSEfilter,2)),'b--','linewidth',2)
ylabel('y');axis([-inf inf,-inf,5]);
legend('Posterior-State RMSE','Observe RMSE')
subplot(3,1,3);
plot(502*dt:dt:5000*dt,RMSEfilter(3,:),'r','linewidth',2);
hold on;
% plot(RMSEforecast(3,:),'b','linewidth',2);
% hold on;
plot(502*dt:dt:5000*dt,repmat(RMSEobs(3),1,size(RMSEfilter,2)),'k--','linewidth',2);
hold on;
% plot(repmat(RMSEFILTER(3),1,size(RMSEfilter,2)),'r--','linewidth',2);
% hold on;
% plot(repmat(RMSEFORE(3),1,size(RMSEfilter,2)),'b--','linewidth',2);
xlabel('Time');ylabel('z');axis([-inf inf,-inf,5]);

end
