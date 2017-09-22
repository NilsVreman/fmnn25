function [lambda, nbr_iters] = linesearch(func, x, d)
%LINESEARCH armijo

alpha = 2;
eps = 1/8;
nbr_iters = 1;

F = @(lambda) func(x+lambda*d);
lambda = 1;

%Checks whether we are outside feasible lambda area (to the right)
while isnan(F(lambda)) || isinf(F(lambda)) || F(lambda) > F(0)
    lambda = lambda/alpha;
    nbr_iters = nbr_iters+1;
end

% Checks whether we are to close to the origin or 
%function is to constant
while F(lambda) < F(0)+max(0.01*abs(F(0)), 10^-4)
    lambda = lambda*alpha;
    nbr_iters = nbr_iters+1;
end

h = 10^-6*lambda;
F_prim_0 = (F(0+h)-F(0))/(h);

if F_prim_0 > 0
    disp('Warning: Positive derivative, set to zero');
    F_prim_0 = 0;
end

T = @(lambda) F(0)+eps*lambda*F_prim_0;

%Here starts the armijo method for a feasible lambda
%calculated above
while F(lambda) < T(lambda)
    lambda = lambda*alpha;
    nbr_iters = nbr_iters+1;
end

while F(lambda) > T(lambda)
    lambda = lambda/alpha;
    nbr_iters = nbr_iters+1;
end

if isnan(func(x+lambda*d)) || func(x+lambda*d)>func(x)
error('Bad job of the line search!')
end

end