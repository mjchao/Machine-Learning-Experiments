function [ dJ ] = costDeriv( thetas , j , X , y )
%COST Calculates the cost of some proposed thetas used to 
% classify some training data
m = size( thetas , 1 );
dJ = 1/m * sum( (sigmoid( X * thetas' ) - y') .* X( : , j ) );
end

