function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
GENERATE_FLAG = 0;
if GENERATE_FLAG == 1
    C_poly = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    sigma_poly = C_poly;
    error = zeros(numel(C_poly)*numel(sigma_poly), 3);
    k = 1;

    for i = 1:numel(C_poly)
        for j = 1:numel(sigma_poly)
            model= svmTrain(X, y, C_poly(i), @(x1, x2) gaussianKernel(x1, x2, sigma_poly(j)));
            pred = svmPredict(model, Xval);
            error(k, 1) = mean(double(pred ~= yval));
            error(k, 2) = i;
            error(k, 3) = j;
            k = k + 1;
        endfor
    endfor

    [tmp, k] = min(error(:, 1));
    C = C_poly(error(k, 2));
    sigma = sigma_poly(error(k, 3));
endif

% =========================================================================

end
