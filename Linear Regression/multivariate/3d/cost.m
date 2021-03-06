function [ J ] = cost( m , data , theta0 , theta1 , theta2 )
%COST calculates the least-squares cost of a hypothesized best fit
x = data( : , : , 1 );
y = data( : , : , 2 );
z = data( : , : , 3 );
squareDiff = (theta0 + theta1 * x + theta2 * y - z).^2;
J = 1/(2*m*m) * sum( squareDiff( : ) );
end

