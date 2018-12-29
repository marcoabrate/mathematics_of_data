% FUNCTION: gout = gradient(forward_operator, adjoint_operator, alpha)
% PURPOSE:  Compute the gradient of fx.
%
function gout = gradient(y, forward_operator, adjoint_operator, x, sz, N)

    y_real = y(1:sz);
    y_imag = y(sz+1:2*sz);
    x_real = x(1:N);
    x_imag = x(N+1:2*N);
    
    gf1 = y_real - real(forward_operator(x_real)) + imag(forward_operator(x_imag));
    gf2 = y_imag - imag(forward_operator(x_real)) - real(forward_operator(x_imag));
    
    gf1_rc = [real(adjoint_operator(gf1)); imag(adjoint_operator(gf1))];
    gf2_rc = [-imag(adjoint_operator(gf2)); real(adjoint_operator(gf2))];
        
    gout = -gf1_rc - gf2_rc;
    
end