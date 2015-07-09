# Machine-Learning-Experiments

Just tinkering around with Machine Learning algorithms. This summer, I'd like to try something with somewhat less-predictable results than what I'm used to programming.

##Directory: Linear Regression
Experiments with attempting to approximate data with linear relationships. This directory deals with single variable linear regression. We produce data from some given linear relationship, y = theta_0 + theta_1 * x, add some noise to it, and apply gradient descent to try and reproduce that relatinship.

###Directory: Linear Regression/Adaptive Learning Rate 
This directory applies single variable gradient descent, but attempt to change the learning rate to make the gradient descent process go faster.

###Directory: Linear Regression/multivariate
This directory deals with multivariate linear regression. We generate data from some given linear relationship, y = theta_0 + theta_1 * x_1 + theta_2 * x_2 + ... + theta_n * x_n, add some noise to it, and apply gradient descent to try and reproduce that relationship.

####Directory: Linear Regression/3d
This directory offers a linear regression example for 3D data only.

####Directory: Linear Regression/generalized
This directory applies linear regression to as many independent variables as one desires, as long as the computing power and memory are available. A run3D function is provided to visually check that the generalized version works with 2 independent variables.
