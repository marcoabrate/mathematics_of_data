addpath('utilities');
clear all

%% Reading image of me
I = imread('IMAGES/image-of-me.jpg');
I = rgb2gray(I);
I = double(I);
m = 256;
disp(m);
I = imresize(I, [m m]);

%% Adding noise to the image
std = 15;
I_noisy = I + std * randn(size(I));

%% Set L1 and TV prox parameters
regularization_parameter_1 = 20;

prox_tv_maxiters            = 1000;
prox_tv_tol                 = 1e-5;
regularization_parameter_tv = 14;

%% Solving L1 and TV prox operators
W = @(x) mdwt(x,daubcqf(8),log2(m));  % From image to wavelet coefficients
WT = @(x) midwt(x,daubcqf(8),log2(m)); % From wavelet coefficients to image
I_prox_1 = WT(proxL1norm(W(I_noisy), regularization_parameter_1));
I_prox_tv = TV_prox(I_noisy, 'lambda', regularization_parameter_tv, 'maxiter', prox_tv_maxiters, 'tol', prox_tv_tol, 'verbose', 0);

%% Computing the PNSRs
PSNR_noisy = PSNR(I, I_noisy);
PSNR_1 = PSNR(I, I_prox_1);
PSNR_tv = PSNR(I, I_prox_tv);

%% Visualizing images

figure,
fontsize = 12;
subplot(141),
imagesc(I), axis image off, colormap gray;
title('Original','fontsize',fontsize,'interpreter','latex');
subplot(142)
imagesc(I_noisy), axis image off;
title(sprintf('Noisy - PSNR = %.2f', PSNR_noisy),'fontsize',fontsize,'interpreter','latex');
subplot(143),
imagesc(I_prox_1), axis image off;
title(sprintf('L1 - PSNR = %.2f', PSNR_1),'fontsize',fontsize,'interpreter','latex');
subplot(144)
imagesc(I_prox_tv), axis image off;
title(sprintf('TV - PSNR = %.2f', PSNR_tv),'fontsize',fontsize,'interpreter','latex');