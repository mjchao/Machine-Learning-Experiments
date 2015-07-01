function [ output_args ] = run( slope , intercept , amount , xMin , xMax , noiseMax )
%run Runs the gradient descent algorithm where we try to adjust the
% learning rate automatically until it stops diverging
dataset = randDataset( slope , intercept , amount , xMin , xMax , noiseMax );
thetas = gradientDescent( dataset );

x = linspace( xMin , xMax );
y = thetas( 1 ) + thetas( 2 ) * x;

hold on;
plot2D( dataset );
plot( x , y );

end

