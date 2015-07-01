function [ newTs ] = gradientDescent( t0 , t1 , alpha , dataset )
%gradientDescent Performs the gradient descent algorithm with the given
% dataset and initial t0 , and t1

derivatives = partialDerivs( t0 , t1 , dataset );
newTs = [ t0 , t1 ] - alpha * derivatives;

end

