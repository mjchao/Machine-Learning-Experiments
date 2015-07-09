function [] = run( thetas , m , noiseSize )
%RUN runs this version of multivariate linear regression for the
%general case
[ x , y ] = genData( thetas , m , noiseSize );

n = size( thetas , 2 );
initialTheta = zeros( 1 , n );
approxThetas = gradientDescent( initialTheta , x , y );
approxThetas
end

