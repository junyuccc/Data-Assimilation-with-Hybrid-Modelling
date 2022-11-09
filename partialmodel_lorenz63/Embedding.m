function [delaytime] = Embedding(timeseries,delays,tau)
%Embedding generate embbedding function
%for getting enough training data
[m,n] = size(timeseries);
delaytime = zeros(m*delays,n-(delays-1)*tau);
    for i = 1:delays
        delaytime((i-1)*m+1:i*m,:) = timeseries(:,(delays-i)*tau+1:end-(i-1)*tau);
    end
end

