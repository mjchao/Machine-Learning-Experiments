function [] = run3D( thetas , m , noiseSize )
%RUN runs this version of multivariate linear regression with 2 
% independent variables. The data points and approximation are graphed
% and shown.
[ x , y ] = genData( thetas , m , noiseSize );
approxThetas = gradientDescent( [ 0 , 0 , 0 ] , x , y );
graph( x , y , approxThetas );
end

