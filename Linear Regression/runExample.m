function [] = runExample( slope , intercept , xMin , xMax , noiseSize )
%runExample runs the gradient descent algorithm and plots the results
dataset = randDataset2DNoise( slope , intercept , 1000 , xMin , xMax , noiseSize );
plotGradientDescent( 0 , 0 , 0.0001 , dataset , 1000 , xMin , xMax );
end

