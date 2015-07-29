function [ ] = plotData( X , y )
%PLOTDATA plots some 2D classification data, where X is a 2D array
% of values and y categorizes each X ordered pair as positive (1) or
% negative (0).
%
% Note that the dimensions of X must be m by 3 because the first column
% of X is always a column of 1s that is the multiplier for theta_0.
hold on;

posX = X( find( y == 1 ) , 2:3 );
scatter( posX( : , 1 ) , posX( : , 2 ) , '+' );

negX = X( find( y == 0 ) , 2:3 );
scatter( negX( : , 1 ) , negX( : , 2 ) , 'o' );

end

