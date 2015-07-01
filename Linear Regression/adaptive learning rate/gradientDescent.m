function [ thetas ] = gradientDescent( dataset )
%GRADIENTDESCENT Summary of this function goes here
%   Detailed explanation goes here
theta0 = 0;
theta1 = 0;
alpha = 0.1;

thetas = [ theta0 , theta1 ];
costBefore = cost( theta0 , theta1 , dataset );
for i = 1:3000
    thetas = gradientDescentIteration( thetas( 1 ) , thetas( 2 ) , alpha , dataset );
    costAfter = cost( thetas( 1 ) , thetas( 2 ) , dataset );
    if ( costAfter > costBefore )
        alpha = alpha / 2;
    end
    costBefore = costAfter;
end

end

