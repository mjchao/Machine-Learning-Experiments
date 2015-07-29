# Machine-Learning-Experiments

Just tinkering around with Machine Learning algorithms. This summer, I'd like to try something with somewhat less-predictable results than what I'm used to programming.

##Directory: Linear Regression
Experiments with attempting to approximate data with linear relationships. This directory deals with single variable linear regression. We produce data from some given linear relationship, y = theta_0 + theta_1 * x, add some noise to it, and apply gradient descent to try and reproduce that relatinship.

Final product: See "Linear Regression/multivariate/generalized"

###Directory: Linear Regression/Adaptive Learning Rate 
This directory applies single variable gradient descent, but attempt to change the learning rate to make the gradient descent process go faster.

###Directory: Linear Regression/multivariate
This directory deals with multivariate linear regression. We generate data from some given linear relationship, y = theta_0 + theta_1 * x_1 + theta_2 * x_2 + ... + theta_n * x_n, add some noise to it, and apply gradient descent to try and reproduce that relationship.

####Directory: Linear Regression/multivariate/3d
This directory offers a linear regression example for 3D data only.

####Directory: Linear Regression/multivariate/generalized
This directory applies linear regression to as many independent variables as one desires, as long as the computing power and memory are available. A run3D function is provided to visually check that the generalized version works with 2 independent variables.

##Directory: Logistic Regression
Experiments with attempting to categorize data into one of two categories: positive or negative. The run function allows you to specify theta_0, theta_1, and theta_2 to generate some random data where sigmoid( theta_0 + theta_1 * x_1 + theta_2 * x_2 ) is the probability that the data point (x_1 , x_2) is positive (theoretically). Then, the run function automatically attempts to draw a line separating these two categories. The probability distribution is actually 0.8 if the sigmoid function yields a value greater than 0.5 and 0.2 otherwise. This makes the data more easily discernable to a viewer. Uncommenting line 22 in genData.m would cause a mess of data points, but the line that the run function produces is still consistent with the actual values of theta.
