function [ output_args ] = graph( dataX , dataY , thetas )
%PLOT3D Plots data in the range [-1 , 1] and the linear approximation 
% of that data, z = theta_0 + theta_1*x + theta_2*y
[ x , y ] = meshgrid( linspace( -1 , 1 , 25 ) , linspace( -1 , 1 , 25 ) );
z = thetas( 1 ) + thetas( 2 )*x + thetas( 3 ) * y;

hold on;
approxSurf = surf( x , y , z );
alpha( approxSurf , .1 );

dataPlot = plot3( dataX( : , 2 ) , dataX( : , 3 ) , dataY , '.' );

view( [-45 45] );
end

