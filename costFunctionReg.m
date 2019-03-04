function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(X*theta);

cs = zeros(m,1);

%for j=1:m
 % cs(j,1) = -y(j,1)*log(h(j,1)) - (1-y(j,1))*log(1-h(j,1));
   
%endfor

  

%sum1 = sum(cs);
%s1 = 0;
%for i = 2:size(X, 2)
 % s1 = s1 + (theta(i,1))^2;
%endfor
%s2 = (lambda*s1)/(2*m);

%J = sum1/m + s2;
  
%s = zeros(size(X, 2),1);

%for i =1:m
  %for j=1:size(X, 2)
  %s(j,1) = s(j,1)+(h(i,1)-y(i,1))*X(i,j);  
 % endfor
  
%endfor
%grad(1,1) = s(1,1)/m;
%for j=2:size(X, 2)
 % grad(j,1) = s(j,1)/m + (lambda*theta(j,1))/m;
%endfor

h = sigmoid(X*theta);
cs = (-y.*log(h) - (1-y).*log(1-h));
cs2 = (lambda/(2*m)).*(theta.*theta);
cs2(1)=0;
sum1 = sum(cs);

J = sum1/m+sum(cs2);
temp = (lambda/m).*theta;
temp(1)=0;
grad = (transpose(X)*(h-y))/m+temp;


% =============================================================

end
