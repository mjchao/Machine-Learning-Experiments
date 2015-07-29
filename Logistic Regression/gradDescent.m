function [ theta ] = gradDescent( initialTheta , X , y , alpha , iterations )
%GRADDESCENT performs gradient descent using the given theta, X, y, and
% learning rate values.
[m , n] = size( initialTheta );

theta = initialTheta;
for iter = 1:iterations
    dThetas = zeros( 1 , n );
    for i = 1:n
        dThetas(i) = -1/m * sum( (sigmoid( X * theta' ) - y') .* X( : , i ) );
    end
    theta = theta + alpha * dThetas;
end
end

