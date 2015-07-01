function [ derivs ] = derivatives( theta0 , theta1 , dataset )
%derivatives Calculates the derivative of the cost function at the given
% theta values

m = size( dataset , 1 );
x = dataset( : , 1 );
y = dataset( : , 2 );

partialT0 = 1 / m * sum ( (theta0 + theta1 * x - y ) );
partialT1 = 1 / m * sum ( (theta0 + theta1 * x - y ) .* x );

derivs = [ partialT0 , partialT1 ];
end

