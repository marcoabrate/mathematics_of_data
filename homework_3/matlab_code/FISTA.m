% FUNCTION: fout = FISTA(fx, gx, gradf, proxg, x0, Lips, maxit, tolx)
% PURPOSE:  FISTA algorithm with non-monotonicity restart.
%
function fout = FISTA(fx, gx, gradf, proxg, x0, Lips, maxit, tolx, verbose)
    
    % defining initial parameters
    x_k = x0;
    x_k_old = x_k;
    theta_k = 1;
    theta_k_old = theta_k;
    fg = zeros(maxit,1);
    
    for k=1:maxit
   
       y_k = x_k + theta_k*(1/theta_k_old - 1)*(x_k - x_k_old);
       x_new = proxg(y_k - (1/Lips)*gradf(y_k), 1/Lips);
       % computing the loss function
       fg(k) = fx(x_new) + gx(x_new);
       
       % restart check (non-monotonicity)
       if k > 1
          if fg(k) > fg(k-1)
              theta_k = 1;
              x_new = proxg(x_k - (1/Lips)*gradf(x_k), 1/Lips);
              fg(k) = fx(x_new) + gx(x_new);
          end
       end
       
       if norm(x_new - x_k) < tolx
           break
       end
       
       if k == 1
          fprintf('First iteration: %f\n', fg(k));
       end
       if verbose && (mod(k,10) == 0 || k < 10)
          fprintf('%d: %f\n', k, fg(k));
       end
          
       % updating the parameters
       x_k_old = x_k;
       x_k = x_new;
       theta_k_old = theta_k;
       theta_k = 0.5*(sqrt(theta_k^4+4*theta_k^2)-theta_k^2);
       
    end
    
    fprintf('Last iteration: %.2f\n', fg(k));
    fout  = x_new;

end
