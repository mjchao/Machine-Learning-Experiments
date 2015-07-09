function [ x , y ] = genData( thetas , m , noiseSize )
%GENDATA generates linear data (y as a function of x) randomly. the data
% has a relationship y = theta_0 * x_0 + theta_1 * x_1 +...+ theta_n * x_n
% + random noise. 
% 
% m is the size of the training data, i.e. how many x's and y's there are.
%
% The number of thetas determines the number of x's. Note that x_0 (which
% has index 1) is always equal to 1
n = size( thetas , 2 );
x = rand( m , n ) * 2 - 1;
x( : , 1 ) = 1;
y = x * thetas';
noise = rand( m , 1 ) * noiseSize;
y = y + noise;
end

