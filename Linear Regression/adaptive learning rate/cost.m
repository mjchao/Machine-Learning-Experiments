function [ c ] = cost( theta0 , theta1 , dataset )
%COST the cost function for the linear regression at the given thetas
m = 1 / size( dataset , 1 );
x = dataset( : , 1 );
y = dataset( : , 2 );
c = 1 / 2 / m * sum ( (theta0 + theta1 .* x - y ) .^ 2 );
end

