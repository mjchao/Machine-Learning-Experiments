function [ y ] = sigmoid( x )
%SIGMOID applies the sigmoid function s(x) = 1/(1+e^-t)

y = 1./(1+exp(-x));


end

