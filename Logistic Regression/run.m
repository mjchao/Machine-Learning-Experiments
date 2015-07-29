function [] = run( thetas , m , noiseSize )
%RUN runs the logistic regression example
[X , y] = genData( thetas , m , noiseSize );
plotData( X , y )

regressionThetas = gradDescent( [ 0 , 1 , -1 ] , X , y , 0.01 , 1000 )
X1 = linspace( -1 , 1 , 100 );
X2 = (-regressionThetas( 1 ) - regressionThetas( 2 ) * X1) / regressionThetas(3);
plot( X1 , X2 );

end

