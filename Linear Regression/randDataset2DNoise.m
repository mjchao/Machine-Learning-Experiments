function [ set ] = randDataset2DNoise( slope , intercept , amount , min , max ,  noiseMax )
%randDataset2DNoise generates a 2D dataset with some noise
    xVals = rand( [amount , 1] ) .* (max - min) + min;
    yVals = intercept + slope .* xVals;
    noise = rand( [amount , 1] ) * 2 * noiseMax - noiseMax;
    set = [ xVals , yVals + noise ];
end

