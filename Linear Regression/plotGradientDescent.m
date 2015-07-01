function [ output_args ] = plotGradientDescent( t0 , t1 , alpha , dataset , iterations , min , max )
%plotGradientDescent Performs the gradient descent algorithm some 
% number of times and then plots the output
hold on;

plot2D( dataset );

thetas = [t0 , t1];
for i = 1:iterations
    thetas = gradientDescent( thetas( 1 ) , thetas( 2 ) , alpha , dataset );
end

thetas
x = linspace( min , max );
y = thetas(1) + x*thetas(2);
plot( x , y );
end

