function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% size_X = size(X)
% size_Theta = size(Theta)

R = (R == 1); % set R to be a 'logical'

all_costs = ((X * Theta') - Y);
costs_squared = all_costs .^ 2;
base_cost = sum(sum(costs_squared(R)));
theta_regularization = lambda * sum(sum(Theta .^ 2));
x_regularization = lambda * sum(sum(X .^ 2));
J = (base_cost + theta_regularization + x_regularization) / 2;


% size_all_costs = size(all_costs)
% size_R = size(R)
% R

X_grad = all_costs .* R * Theta + lambda * X;

% @TODO Find an equally elegant (as X_grad above) vectorized solution.
for j = 1:num_users
    for i = 1:num_movies
        if R(i, j) != 1
            continue
        end
        Theta_grad(j, :) += (Theta(j, :) * X(i, :)' - Y(i, j)) * X(i, :);
    end
end

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
















% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
