function D_new = updateD(y, y_new, f, D, method)
%UPDATED updates the matrix D in the DFP algorithm

p = y_new - y;
q = grad(f,y_new)-grad(f,y);
    
if strcmp(method, 'DFP')
    D_new = D + 1/(p'*q)*(p*p')-1/(q'*D*q)*(D*q*q'*D);
else %Then 'BFGS' method
    D_new = D+(1+(q'*D*q)/(p'*q))*1/(p'*q)*(p*p')-1/(p'*q)*(p*q'*D+D*q*p');
end

end
