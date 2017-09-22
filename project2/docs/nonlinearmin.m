function nonlinearmin(func, start, method, tol, printout)
%NONLINEARMIN
%Description coming up
tic;
method = upper(method); %Changes all char to upper char

if strcmp(method, 'DFP') || strcmp(method, 'BFGS')
    x = start;
    termination = false;
    counter = 0;
    print_data(0, 0, 0, 0, 0, 0, printout, 1)
    
    while termination == false
        counter = counter+1;
        total_ls_iters = 0;
        y = x;
        D = eye(length(start));
                
        for j = 1:length(start) %length(start) = n Because for quadratic function you need n inner iterations to solve the problem exact
            d = -D*grad(func,y);
            
            if norm(d) == 0
                lambda = 0;
                break;
            end
            
            [lambda, nbr_iters] = linesearch(func, y, d);
            total_ls_iters = total_ls_iters+nbr_iters;
            y_new = y + lambda*d;
            D = updateD(y, y_new, func, D, method);
            
            if isnan(D)
                break;
            end
            
            y = y_new;
        end
        
        x_new = y;
        termination = (norm(x_new-x)<tol)*(abs(func(x_new)-func(x))<tol);
            %True if relative function increase is smaller than tol
        step_size = norm(x_new-x);
        x = x_new;
        
        print_data(counter, x, step_size, func, total_ls_iters, lambda, printout, 0);
    end
    
else
    fprintf('Please choose either the DFP or the BFGS method.\n')
end

toc;
end

