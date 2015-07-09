function [ solution ] = gradientDescent( thetas , x , y )
%GRADIENTDESCENT Performs the gradient descent algorithm
alpha = 1;
m = size( x , 1 );
n = size( x , 2 );

for i = 1:1000
    cost = x * thetas' - y;
    dThetas = [];
    for j = 1:n
         dThetas( 1 , j ) = -alpha * 1/m * sum( cost .* x( : , j ) );
    end
    thetas = thetas + dThetas;
end
solution = thetas;
end

