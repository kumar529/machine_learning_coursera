function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
I = eye(num_labels);
Y = zeros(m, num_labels); % m * k
for i=1:m
	Y(i, :) = I(y(i), :);
end

A1 = [ones(m, 1) X]; % m * (n + 1)
Z2 = A1 * Theta1'; % ( m * (n + 1) ) * ( (n + 1) * l) => m * l
A2 = sigmoid(Z2);
A2 = [ones(size(Z2, 1), 1) A2]; % (m * (l + 1))
Z3 = A2 * Theta2'; % (m * (l + 1)) * ((l + 1) * k) => m * k
H = A3 = sigmoid(Z3); % m * k

J = -1/m * sum(sum(Y.*log(H) + (1-Y).*log(1-H)), 2);

regVal = (lambda/(2*m)) * (sum(sum(Theta1(:, 2:end).^2), 2) + sum(sum(Theta2(:, 2:end).^2), 2));

J = J + regVal;

gradCost1 = zeros(hidden_layer_size, input_layer_size + 1);
gradCost2 = zeros(num_labels, hidden_layer_size + 1);

for t = 1:m
	a1 = [1, X(t, :)]'; % (n + 1) * 1
	z2 = Theta1 * a1; % l * 1
	a2 = [1; sigmoid(z2)]; % (l + 1) * 1
	z3 = Theta2 * a2; % k * 1
	a3 = sigmoid(z3); % k * 1
	y = Y(t, :)'; % k * 1
	delta3 = a3 - y; % k * 1
	delta2 = Theta2' * delta3 .* (a2 .* (1 - a2)); % (l + 1) * 1
	delta2 = delta2(2:end, :); % l * 1
	gradCost2 = gradCost2 + delta3 * a2'; % k * (l + 1)
	gradCost1 = gradCost1 + delta2 * a1'; % l * (n + 1)
end

Theta1_grad = (1/m) * gradCost1; % l * (n + 1)
Theta2_grad = (1/m) * gradCost2; % k * (l + 1)

reg1 = (lambda/m) * Theta1; % l * (n + 1)
reg1(:, 1) = 0;
reg2 = (lambda/m) * Theta2; % k * (l + 1)
reg2(:, 1) = 0;

Theta1_grad = Theta1_grad + reg1;
Theta2_grad = Theta2_grad + reg2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
