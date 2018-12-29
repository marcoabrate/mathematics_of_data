% FUNCTION: gout = gradient19(forward_operator, adjoint_operator, alpha)
% PURPOSE:  Compute the gradient of fx (for problem 4 equation 19).
%
function gout = gradient19(y, forward_operator, adjoint_operator, x)
        
    gout = -adjoint_operator(y - forward_operator(x));
    
end