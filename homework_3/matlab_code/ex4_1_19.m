clear all
%% Problem size - image side
%addpath('export_fig');
addpath('utilities/');
I = imread('IMAGES/image-of-me.jpg');
I = rgb2gray(I);
I = double(I);
m = 1024;
N = m^2;
I = imresize(I, [m m]);

rate = 0.4;
p = floor(rate*N);

ind_mat = zeros(m, m);
%% Random indices
ind = randperm(N, p); % p random indices
ind_mat(ind) = 1;

I_subsampled = ind_mat.*I;
PSNR_subsampled = PSNR(I, I_subsampled);


 %% Define operators

% Measurement operators
measurement_forward             = @(x) ind_mat.*x;   % P_Omega
measurement_backward            = @(x) ind_mat.*x;  % P_Omega ^T

% Define the overall operator
forward_operator                = @(x) measurement_forward(x); % A
adjoint_operator                = @(x) measurement_backward(x); % A^T

%% Defining parameters
Lips    = sqrt(2);

% Generate measurements on complex domain and transfer to real domain
y = measurement_forward(I);

% Optimization parameters
maxit = 100;
tolx  = 1e-5;

%% WAVELET RECOVERY
% regularization parameter
regularization_parameter_tv  = 1.3;
% Initial point
x0      = zeros(m, m);

fx      = @(x) 0.5 * norm(y - forward_operator(x), 'fro')^2;
gx      = @(x) regularization_parameter_tv * TV_norm(x, 'iso');
proxg   = @(x, gamma) TV_prox(x, 'lambda', gamma*regularization_parameter_tv, ...
    'maxiter', maxit/2, 'tol', tolx*1e4, 'verbose', 0);

gradf = @(x) gradient19(y, forward_operator, adjoint_operator, x);

% applying FISTA
time_rec = tic;
I_rec = FISTA19(fx, gx, gradf, proxg, x0, Lips, maxit, tolx, 1);
time_rec = toc(time_rec);


%% VISUALIZING results
PSNR_tv = PSNR(I, I_rec);

figure,
fontsize = 12;
subplot(131),
imagesc(I), axis image off, colormap gray;
title('Original','fontsize',fontsize,'interpreter','latex');
subplot(132)
imagesc(I_subsampled), axis image off, colormap gray;
title(sprintf('Subsampled - PSNR = %.2f', PSNR_subsampled),'fontsize',fontsize,'interpreter','latex');
subplot(133),
imagesc(I_rec), axis image off, colormap gray;
title(sprintf('TV (%f s) - PSNR = %.2f', time_rec, PSNR_tv),'fontsize',fontsize,'interpreter','latex');