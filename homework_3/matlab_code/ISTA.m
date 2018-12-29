% FUNCTION: iout = ISTA(fx, gx, gradf, proxg, x0, Lips, maxit, tolx, F_star, verbose)
% PURPOSE:  ISTA algorithm.
%
function iout = ISTA(fx, gx, gradf, proxg, x0, Lips, maxit, tolx, F_star, verbose)
    
    % defining initial parameters
    x_k = x0;
    fg_pl = zeros(maxit,1);
    
    for k=1:maxit
   
       x_k = proxg(x_k - (1/Lips)*gradf(x_k), 1/Lips);
       
       fg_pl(k) = abs(fx(x_k) + gx(x_k) - F_star) / F_star;
       
       if fg_pl(k) < tolx
           break
       end
       
       if k == 1
          fprintf('First iteration: %f\n', fg_pl(k));
       end
       if verbose && (mod(k,10) == 0 || k < 10)
          fprintf('%d: %f\n', k, fg_pl(k));
       end
       
    end
    
    fprintf('Last iteration: %f\n', fg_pl(k));
    iout  = x_k;
    
    figure,
    semilogy(fg_pl);
    grid on;
    title("ISTA");
    xlabel("iteration k");
    ylabel("log((F(x_k) - F_{true})/F_{true})");

end