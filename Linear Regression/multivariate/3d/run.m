function [] = run( m , theta0 , theta1 , theta2 , noise )
%RUN randomly generates some 3D and applies gradient descent to try
% and fit that data

minX = 0;
maxX = 10;
minY = 0;
maxY = 10;

data = genData3D( m , theta0 , theta1 , theta2 , minX , maxX , minY , maxY , noise );

learningRate = 0.01;
thetas = [ 0 , 1 , 1 ];
for i = 1:10000
    thetas = gradientDescent( data , m , learningRate , thetas );
end

[x y] = meshgrid( linspace(minX , maxX , m) , linspace(minY , maxY , m) );
z = thetas(1) + thetas(2)*x + thetas(3)*y;

hold on;

dataX = data( : , : , 1 );
dataY = data( : , : , 2 );
dataZ = data( : , : , 3 );
s1 = surf( dataX , dataY , dataZ );
alpha( s1 , .4 );

s2 = surf( x , y , z );
alpha( s2 , 1 );

view( [45 , 45] );
thetas
end

