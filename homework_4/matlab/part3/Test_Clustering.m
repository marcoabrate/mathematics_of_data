% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % EE556 - MATHMATICS OF DATA  % % % % % % % % % % % % FALL 2018 % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% k-means clustering BY CONVEX PROGRAMMING % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% PURPOSE: We will solve the following problem formulation with % % % % % %
% HomotopyCGM method.                                    
%                   min_x  <C, X>        
%                   s.t.:  A(X) = b,
%                          X => 0,
%                          norm_nuc(x) <= kappa, X is PSD
%
% Prepared by Alp Yurtsever & Mehmet Fatih Sahin
% Laboratory for Information and Inference Systems (LIONS)
% Ecole Polytechnique Federale de Lausanne (EPFL) - SWITZERLAND
% Last modification: December 5, 2018
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

%% Clear varibles, add dependencies, etc.
clearvars;
rng(1);


%% Problem Construction
load clustering_data
C = Problem.C; % Distance Matrix
N = Problem.N;
k = Problem.k;
opt_val = Problem.opt_val;
clearvars Problem
%% Define operators
% We provide 4 operators:
% 1- A1: Linear operator that takes the row sums
% 2- At2: Conjugate of operator A1
% 3- A2: Linear operator that takes the column sums 
% 4- At2: Conjugate of operator A2
A1 = @(x) sum(x,2);
At1 = @(y) repmat(y,[1,N]);
A2 = @(x) sum(x,1);
At2 = @(y) repmat(y,[N,1]);
% Define the feasible singleton
b1 = ones(N,1);
b2 = ones(1,N);

%%
Aoper.A1 = A1;
Aoper.A2 = A2;

AToper.At1 = At1;
AToper.At2 = At2;

b.b1 = b1;
b.b2 = b2;
%% Run HomotopyCGM method
maxit = 1e4;
kappa = k; % k is the number of clusters
beta0 = 1; % tuning parameter
[xHCGM, outHCGM] = HomotopyCGM(C, Aoper, AToper, b, N, kappa, maxit, beta0);

%% NOTE: This experiment is based on the theory and the codes publised in
% 'Blind Deconvolution using Convex Programming' by A.Ahmed, B.Recht and
% J.Romberg.
