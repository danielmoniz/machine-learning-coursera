function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
thetaX = X * theta;
h = sigmoid(thetaX);
extra_reg_cost = (lambda / (2*m)) * sum(theta .^ 2);
cost_sum = sum((-y * log(h)') - (1 - y) * (log(1 - h)'));
J = (1/m) * cost_sum + extra_reg_cost;

% grad = zeros(size(theta));
extra_reg_grad = (lambda / m) * theta;
grad = (1 / m) * (X' * (h - y)) + extra_reg_grad;

grad(1, 1) = (1 / m) * (X(:, 1)' * (h - y));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
