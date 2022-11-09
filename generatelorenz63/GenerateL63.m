function [truth,obs] = GenerateL63(T,N,dt,Q,R)
    truth = zeros(N,T);     %%% noise free 'truth'
    obs = zeros(N,T);       %%% noisy observations
    
    transient = 1000;
    substeps = ceil(dt/.05);
    h = dt/substeps;
    state = rand(N,1);
    %%% Compute the matrix square roots of Q and R, used to generate
    %%% appropriately correlated noise samples below
    [U,S,~] = svd(Q);
    rootQ = U*diag(sqrt(diag(S)))*U';
    [U,S,~] = svd(R);
    rootR = U*diag(sqrt(diag(S)))*U';
    
    %%% Run an initial transient to get onto the attractor
    for i = 1:transient
        state = L63Dynamics(state,dt);
    end
    
    for i = 1:T
        for k = 1:substeps
            %%% RK4
            k1=h*Lorenz63_diff(state);
            k2=h*Lorenz63_diff(state+k1/2);
            k3=h*Lorenz63_diff(state+k2/2);
            k4=h*Lorenz63_diff(state+k3);
            state=state+k1/6+k2/3+k3/3+k4/6;
            
            %%% Stochastic forcing with coviariance Q
            state = state+sqrt(h)*rootQ*randn(N,1);
%             state = state+rootQ*randn(N,1);
        end
        %%% Clean 'true' state
        truth(:,i) = state;
        %%% Noisy observed state with noise covariance R
        obs(:,i) = state+rootR*randn(N,1);
    end   
end

