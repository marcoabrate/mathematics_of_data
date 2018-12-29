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

%% Wavelet operators
% Define the function handles that compute
% the products by W (DWT) and W' (inverse DWT)
W = @(x) mdwt(x,daubcqf(8),log2(m));  % From image to wavelet coefficients
WT = @(x) midwt(x,daubcqf(8),log2(m));


 %% Define operators

% Vectorized transformations
representation_operator         = @(x) W(x);
representation_operator_trans   = @(x) WT(x);

% Measurement operators
measurement_forward             = @(x) ind_mat.*x;   % P_Omega
measurement_backward            = @(x) ind_mat.*x;  % P_Omega ^T

% Define the overall operator
forward_operator                = @(x) measurement_forward(representation_operator_trans(x)); % A
adjoint_operator                = @(x) representation_operator(measurement_backward(x)); % A^T

%% Defining parameters
Lips    = sqrt(2);

% Generate measurements on complex domain and transfer to real domain
y = measurement_forward(I);

% Optimization parameters
maxit = 100;
tolx  = 1e-5;

%% WAVELET RECOVERY
% regularization parameter
regularization_parameter_lasso  = 5;
% Initial point
x0      = zeros(m, m);

fx      = @(x) 0.5 * norm(y - forward_operator(x), 'fro')^2;
gx      = @(x) regularization_parameter_lasso * norm(reshape(x, [N,1]), 1);
proxg   = @(x, gamma) proxL1norm(x, regularization_parameter_lasso*gamma);

gradf = @(x) gradient18(y, forward_operator, adjoint_operator, x);

% applying FISTA
time_wav = tic;
I_wav = FISTA18(fx, gx, gradf, proxg, x0, Lips, maxit, tolx, 1);
time_wav = toc(time_wav);


%% VISUALIZING results
I_rec = representation_operator_trans(I_wav);
PSNR_1 = PSNR(I, I_rec);

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
title(sprintf('L1 (%f s) - PSNR = %.2f', time_wav, PSNR_1),'fontsize',fontsize,'interpreter','latex');