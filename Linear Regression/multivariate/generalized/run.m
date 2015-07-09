function [ approxThetas ] = run( thetas , m , noiseSize )
%RUN Runs this version of multivariate linear regression for the
% general case. It generates data points using a given linear
% relationship, applies some noise to it, then uses multivariate
% linear regression to try to determine the origin linear relation.
%
% thetas are the real coefficients of some relation:
% y = theta0*x_0 + theta_1*x_1 + theta_2*x_2 + ... + theta_n*x_n
% where x_0, ..., x_n are independent variables and x_0 = 1 always
%
% m is the number of data points to generate. More data points will
% probably yield better regression results
%
% noiseSize is the magnitude of the noise to add to the datapoints.
[ x , y ] = genData( thetas , m , noiseSize );

n = size( thetas , 2 );
initialTheta = zeros( 1 , n );
approxThetas = gradientDescent( initialTheta , x , y );
end

