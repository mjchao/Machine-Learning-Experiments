function [ data ] = genData3D( m , intercept , xCoeff , yCoeff , lowX , highX , lowY , highY , maxNoise )
%genData3D generates m random entries (x_i, y_i, z_i) 
[x , y] = meshgrid( linspace( lowX , highX , m ) , linspace( lowY , highY , m ) );
noise = rand( [ m , m ] ) * 2 * maxNoise - maxNoise;
z = intercept + (xCoeff * x) + (yCoeff * y) + noise;
data( : , : , 1 ) = x;
data( : , : , 2 ) = y;
data( : , : , 3 ) = z;
end

