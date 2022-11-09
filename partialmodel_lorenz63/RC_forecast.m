function [forecast_x,rt] = RC_forecast(rt,x_unscented,Win,Wres,Wout,resSize,gamma)
%RC_FORECAST: updating the state using the trained RC  
%   此处显示详细说明
[m,n] = size(x_unscented);
forecast_x = zeros(m,n);
rr=zeros(resSize,n);
    for i=1:n
        r=rt;
        ut = x_unscented(:,i); 
        r = (1-gamma)*r + gamma*tanh(Win*ut+Wres*r);
        r1 = r;
        r1(resSize/2+1:end) = r1(resSize/2+1:end).^2;
        forecast_x(:,i) = Wout * r1;
        rr(:,i)=r;
    end
    rt=rr(:,1);
end

