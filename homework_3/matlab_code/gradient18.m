% FUNCTION: gout = gradient18(forward_operator, adjoint_operator, alpha)
% PURPOSE:  Compute the gradient of fx (for problem 4 equation 18).
%
function gout = gradient18(y, forward_operator, adjoint_operator, x)
        
    gout = -adjoint_operator(y - forward_operator(x));
    
end