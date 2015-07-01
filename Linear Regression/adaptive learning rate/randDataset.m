function [ set ] = randDataset( slope , intercept , amount , min , max ,  noiseMax )
%randDataset2DNoise Generates a 2D dataset with some noise. this is the
% dataset to which we'll apply linear regression and see if we can come
% up with a good approximation.
    xVals = rand( [amount , 1] ) .* (max - min) + min;
    yVals = intercept + slope .* xVals;
    noise = rand( [amount , 1] ) * 2 * noiseMax - noiseMax;
    set = [ xVals , yVals + noise ];
end

