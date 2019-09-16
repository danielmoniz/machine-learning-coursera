function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
m = size(X, 1);

% You need to return the following variables correctly.
idx = zeros(m, 1);

for i = 1:m
    x = X(i, :);
    closest_centroid = 0;
    shortest_distance = -1;
    for k = 1:K
        centroid = centroids(k, :);
        distance = norm(x - centroid) ^ 2;
        if shortest_distance == -1 || distance < shortest_distance
            closest_centroid = k;
            shortest_distance = distance;
        end
    end
    idx(i) = closest_centroid;
end

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%







% =============================================================

end

