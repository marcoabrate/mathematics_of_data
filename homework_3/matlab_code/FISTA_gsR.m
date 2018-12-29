% FUNCTION: fout = FISTA_gsR(fx, gx, gradf, proxg, x0, Lips, maxit, tolx, F_star, verbose)
% PURPOSE:  FISTA (for problem 4 equation 18) with gradient scheme restart algorithm.
%
function iout = FISTA_gsR(fx, gx, gradf, proxg, x0, Lips, maxit, tolx, F_star, verbose)
    
    % defining initial parameters
    x_k = x0;
    N = size(x0,1)^2;
    x_k_old = x_k;
    theta_k = 1;
    theta_k_old = theta_k;
    fg_pl = zeros(maxit,1);
    
    for k=1:maxit
   
       y_k = x_k + theta_k*(1/theta_k_old - 1)*(x_k - x_k_old);
       x_new = proxg(y_k - (1/Lips)*gradf(y_k), 1/Lips);
       
       fg_pl(k) = abs(fx(x_k) + gx(x_k) - F_star) / F_star;
       
       if fg_pl(k) < tolx
           break
       end
       
       % restart check (gradient scheme)
       if k > 1
          if dot(reshape(y_k, [N,1]) - reshape(x_new, [N,1]), ...
                  reshape(x_new, [N,1]) - reshape(x_k, [N,1])) > 0
              theta_k = 1;
              x_new = proxg(x_k - (1/Lips)*gradf(x_k), 1/Lips);
          end
       end
       
       if k == 1
          fprintf('First iteration: %f\n', fg_pl(k));
       end
       if verbose && (mod(k,10) == 0 || k < 10)
          fprintf('%d: %f\n', k, fg_pl(k));
       end
          
       % updating the parameters
       x_k_old = x_k;
       x_k = x_new;
       theta_k_old = theta_k;
       theta_k = 0.5*(sqrt(theta_k^4+4*theta_k^2)-theta_k^2);
       
    end
    
    fprintf('Last iteration: %f\n', fg_pl(k));
    iout  = x_new;
    
    figure,
    semilogy(fg_pl);
    grid on;
    title("FISTA with gradient scheme restart");
    xlabel("iteration k");
    ylabel("log((F(x_k) - F_{true})/F_{true})");

end