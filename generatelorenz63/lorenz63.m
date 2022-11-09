clear;clc;close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Generate Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = 3;                         %%% number of state variables
T = 30000;                       %%% total number of time steps
dt = 0.01;                      %%% time between observations
dynnoiseVariance = 0;           %%% system noise (stochastic forcing)
obsnoiseVariance = 16;           %%% observation noise
Q = dynnoiseVariance*eye(N);    %%% system noise covariance matrix
R = obsnoiseVariance*eye(N);    %%% obs noise covariance matrix

[truth,obs] = GenerateL63(T,N,dt,Q,R);
save lorenz63_0_16 truth obs