function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  % Return:
  %   f: value of cost function
  %   g: value of gradient
  m=size(X,2); %number of examples
  n=size(X,1); %dimension of example,theta
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
 
  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  for j=1:m
      p=1./(1+e.^(-theta'*X(:,j)));
      r=y(j)*log(p) + (1-y(j))*log(1-p);
      f=f+r;
  end
  f=-f;
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
  p=1./(1+e.^(-theta'*X));
  diff=p-y;
  for i=1:n
      g(i)=X(i,:)*diff';
  end
