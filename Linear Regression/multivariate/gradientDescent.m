function [ thetas ] = gradientDescent( data , m , alpha , thetas )
%GRADIENTDESCENT performs 1 iteration of gradient descent
x = data( : , : , 1 );
x = x(:);
y = data( : , : , 2 );
y = y(:);
z = data( : , : , 3 );
z = z(:);
theta0 = thetas( 1 );
theta1 = thetas( 2 );
theta2 = thetas( 3 );

differences = theta0 + theta1 * x + theta2 * y - z;
dTheta0 = alpha * 1/(m*m) * sum( differences );
dTheta1 = alpha * 1/(m*m) * sum( differences .* x );
dTheta2 = alpha * 1/(m*m) * sum( differences .* y );
thetas = [ theta0 - dTheta0 , theta1 - dTheta1 , theta2 - dTheta2 ];
end

