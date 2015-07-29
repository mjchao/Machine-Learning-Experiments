function [ X y ] = genData( thetas , m , noiseSize )
%GENDATA Generates data for a classification problem. The probability of
% some element being positive is 1/(1+e^-(t(X))) where
% t(X) = theta_0 + theta_1*x_1 + ... + theta_n*x_n
%
% m determines the size of the training date
%
% noiseSize applies some noise to the data

n = size( thetas , 2 );
X = rand( m , n ) * 2 - 1;
X( : , 1 ) = 1;
noise = rand( m , 1 ) * noiseSize * 2 - noiseSize ;
probabilities = sigmoid( X * thetas' + noise );
for i = 1:m
    if ( probabilities( i ) > 0.5 )
        distribution = [ 0.1 , 0.9 ];
    else
        distribution = [ 0.9 , 0.1 ];
    end
    y(i) = randsample( [0,1] , 1 , true , distribution );
    %y(i) = randsample( [0,1] , 1 , true , [ 1-probabilities( i ) ,  probabilities( i ) ] );
end
end

