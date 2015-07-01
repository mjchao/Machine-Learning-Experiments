function [ newThetas ] = gradientDescentIteration( theta0 , theta1 , alpha , dataset )
%gradientDescentIteration Performs one iteration of the gradient descent
% algorithm

derivs = derivatives( theta0 , theta1 , dataset );
newThetas = [ theta0 , theta1 ] - alpha * derivs;

end

