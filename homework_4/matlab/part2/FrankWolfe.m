function [X, fx] = FrankWolfe( Aoper,AToper,b,n1,n2,kappa,maxit,plotFunc )
% PURPOSE: We will solve the following problem formulation with
% Frank-Wolfe's method.                                    
%                   min_x  0.5*norm(A(x) - b)^2        
%                   s.t.:  norm_nuc(x) <= kappa,     
%
% Laboratory for Information and Inference Systems (LIONS)
% Ecole Polytechnique Federale de Lausanne (EPFL) - SWITZERLAND
% Last modification: November 23, 2017

% Print the caption
fprintf('%s\n', repmat('*', 1, 22));
fprintf(' Frank-Wolfe''s method \n');
fprintf('%s\n', repmat('*', 1, 22));
fprintf('  Iter|  Objective \n');
fprintf('%s\n', repmat('*', 1, 22));

% Initialize
AX_t = 0;   % zeros
X = 0;      % zeros

% Options for the subsolver
optsProPack.tol = 1e-4;     % error tolerance

% keep track of objective value
fx = nan(maxit+1,1);

% The main loop
for iter = 0:maxit
    
    % Print the objective values ...
    fx(iter+1) = 0.5*norm(AX_t - b,2)^2;
    fprintf('%6d| %5.4e\n', iter, fx(iter+1));
    
    % Form the residual and fix the operator to be used in svds. 
    res_cur = AX_t - b;
    eigsAToper1 = @(w)AToper.matvec(???, w);
    eigsAToper2 = @(w)AToper.rmatvec(???, w);
    [topLe_vec, ~ , topRe_vec] = lansvd(eigsAToper1, eigsAToper2, n2, n1, 1, 'L', optsProPack);
    % Note: we could also used svds. Lansvd and svds solve the same problem with similar
    % but different approaches. Svds in older versions of Matlab does not accept function
    % handles as inputs, this is why we rather used lansvd here. If you run into trouble
    % with lansvd on your computer, try to use svds (with properly modifying the inputs)
    
    % Apply A to the rank 1 update
    AXsharp_t = Aoper(topLe_vec, -kappa, topRe_vec);
    
    % Step size
    weight  = ???;
        
    % Update X
    X       = ???;
    
    % Update A*X efficiently using linearity
    AX_t    = (1 - weight)*AX_t + weight*AXsharp_t;

    % Show the reconstruction (at each 10 iteration)
    if floor(iter/10) == iter/10
        figure(101)
        [U,~,~] = svd(X,0);
        plotFunc(U(:,1))
    end
    
end

%% NOTE: This experiment is based on the theory and the codes publised in
% 'Blind Deconvolution using Convex Programming' by A.Ahmed, B.Recht and
% J.Romberg.
