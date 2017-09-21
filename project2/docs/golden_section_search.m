function [a_new, b_new] = golden_section_search(a, b, n, F)
%GOLDEN_SECTION_SEARCH
% a = min_interval
% b = max_interval
% n = nbr_iterations
% F = F(lambda) = f(x + lambda*d) , x, d = vectors

alpha = (sqrt(5)-1)/2;
a_k = a;
b_k = b;
lambda_k = a_k + (1 - alpha)*(b_k - a_k);
mu_k = a_k + alpha*(b_k - a_k);

F_lambda = F(lambda_k);
F_mu = F(mu_k);

for i = 1:n
    
    if F_lambda < F_mu
        b_k = mu_k;
        mu_k = lambda_k;
        lambda_k = a_k + (1 - alpha)*(b_k - a_k);
        F_mu = F_lambda;
        F_lambda = F(lambda_k)
    else
        a_k = lambda_k;
        lambda_k = mu_k;
        mu_k = a_k + alpha*(b_k - a_k);
        F_lambda = F_mu;
        F_mu = F(mu_k);
    end
end

a_new = a_k;
b_new = b_k;

end