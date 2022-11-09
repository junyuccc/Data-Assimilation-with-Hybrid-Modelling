function [x_unscented,weights] = unscented(xa,P,c,n)
%UNSCENTED generate unscented ensemble of state x
%   此处显示详细说明
[U,S,~] = svd(P);
rootP = U*diag(sqrt(diag(S)))*U';
x_unscented = zeros(n,2*n+1);
x_unscented(:,1) = xa;
x_unscented(:,2:n+1) = repmat(xa,1,n) + c*rootP;
x_unscented(:,n+2:2*n+1) = repmat(xa,1,n) - c*rootP;
weights = [c.^2-n ones(1,2*n)/2]/(c.^2);
end

