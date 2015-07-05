function [] = plot3D( data )
%PLOT3D plots 3 dimensional data
x = data( : , : , 1 );
y = data( : , : , 2 );
z = data( : , : , 3 );
surf( x , y , z );
end

