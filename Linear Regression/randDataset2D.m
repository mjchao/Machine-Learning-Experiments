function [ set ] = randDataset2D( slope , intercept , amount )
% randDataset2D generates a linear 2D dataset with a fixed amount of noise
    xVals = rand( [ amount , 1 ] ) .* 100;
    yVals = intercept + slope*xVals;
    noise = rand( [ amount , 1 ] ) .* 25;
    set = [ xVals , yVals + noise ];
end

