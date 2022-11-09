function state = L63Dynamics(state,dt)

    substeps = ceil(dt/.05);
    h = dt/substeps;
    
    for k = 1:substeps
        %%% RK4
        k1=h*Lorenz63_diff(state);
        k2=h*Lorenz63_diff(state+k1/2);
        k3=h*Lorenz63_diff(state+k2/2);
        k4=h*Lorenz63_diff(state+k3);
        state=state+k1/6+k2/3+k3/3+k4/6;
    end
end

