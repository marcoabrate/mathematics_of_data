function [X, output] = HomotopyCGM(C, Aoper, AToper, b, n, kappa, maxit, beta0)
% PURPOSE: We will solve the following problem formulation with
% HomotopyCGM method.
%                   min_x  <C, X>
%                   s.t.:  A(X) = b,
%                          X => 0,
%                          norm_nuc(x) <= kappa, X is PSD
%
%
% Prepared by Alp Yurtsever & Mehmet Fatih Sahin
% Laboratory for Information and Inference Systems (LIONS)
% Ecole Polytechnique Federale de Lausanne (EPFL) - SWITZERLAND
% Last modification: December 5, 2018

% Print the caption
fprintf('%s\n', repmat('*', 1, 22));
fprintf(' HomotopyCGM method \n');
fprintf('%s\n', repmat('*', 1, 22));
% fprintf('  Iter|  Objective \n');
% fprintf('%s\n', repmat('*', 1, 22));

% Initialize
X = zeros(n,n);      % zeros
AX1_b = zeros(size(b.b1));


% Options for the subsolver
OPTS.issym = 1;
OPTS.isreal = 1;
OPTS.tol = 1e-9;

% keep track of objective value and feasibility gap
output.feasibility1 = [];  % norm(A1(X)-b1)/norm(b1)
output.feasibility2 = [];  % dist(X, \mathcal{K})
output.objective = [];     % f(X)
output.iteration = [];


% Forward and backward operators
A1 = Aoper.A1;
At1 = AToper.At1;

A2 = Aoper.A2;
At2 = AToper.At2;

b1 = b.b1;
b2 = b.b2;

C = double(C);
% The main loop
for iter = 1:maxit
    
    % Update step size
    gamma = ???
    
    % Update beta
    beta = ???
    
    
    % Write down the vk to use in the lmo (eigenvalue routine)
    v_k = ???
    v_k = 0.5*(v_k + v_k');
    
    % Linear minimization oracle
    % Use eigs routine with proper settings to calculate the lmo
    % Type 'doc eigsh' to the command line for detailed explanation of
    % function
    [u,s] = eigs(???,OPTS);
    u = sqrt(kappa)*u;
    X_bar = u*u';
    
    % For warmstart in the next iteration
    OPTS.v0 = u;
    
    % Obtain A*X_bar - b
    AX_bar_b = A1(X_bar)-b1;
    
    % Update A*X - b
    AX1_b = (1-gamma)*AX1_b + gamma*(AX_bar_b);
    
    % Update X
    X = ????
    
    % Print iterations
    printIter()    
end

    function [] = printIter()
        if any(iter == unique(ceil(1.1.^[0:150]))) || iter == maxit || iter == 1 || iter == 1e5
            output.iteration(end+1,1) = iter;
            output.feasibility1(end+1,1) = norm(AX1_b)/norm(b1);
            output.feasibility2(end+1,1) = norm(min(X,0),'fro');
            output.objective(end+1,1) = C(:)'*X(:);

            fprintf('\n%5d | %5.4e | %5.4e | %5.4e | %5.4e |\n\n', ...
                iter, output.objective(end), output.feasibility1(end), ...
                output.feasibility2(end));            
        end
    end
end
%%