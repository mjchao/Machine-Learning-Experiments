function [ derivs ] = partialDerivs( t0 , t1 , dataset )
%partialT0 Takes the partial derivative of the cost function
% with respect to theta_0

x = dataset( : , 1 );
y = dataset( : , 2 );
m = size( dataset , 1 );
partialT0 = 1/m * sum( (t0 + t1.*x) - y );
partialT1 = 1/m * sum( ((t0 + t1.*x) - y).*x );
derivs = [ partialT0 , partialT1 ];
end

