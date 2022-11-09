function ydot = Lorenz63_diff(state)
    ydot = zeros(size(state));
    ydot(1,:) = 10*(state(2,:)-state(1,:));
    ydot(2,:) = state(1,:)*(28-state(3,:))-state(2,:);
    ydot(3,:) = state(1,:)*state(2,:)-8/3*state(3,:);      
end

