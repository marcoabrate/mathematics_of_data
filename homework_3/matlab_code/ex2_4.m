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

%% Solving the L1 denoising problem on a certain range of the regularization parameter
W = @(x) mdwt(x,daubcqf(8),log2(m));  % From image to wavelet coefficients
WT = @(x) midwt(x,daubcqf(8),log2(m)); % From wavelet coefficients to image

regularization_parameters_1 = linspace(10, 30, 20);
PSNR_1 = zeros(size(regularization_parameters_1));
for idx = 1:numel(regularization_parameters_1)
    lambda = regularization_parameters_1(idx);
    I_prox_1 = WT(proxL1norm(W(I_noisy), lambda));
    PSNR_1(idx) = PSNR(I, I_prox_1);
end

%% Set TV prox parameters
prox_tv_maxiters            = 100;
prox_tv_tol                 = 1e-5;

%% Solving the TV denoising problem on a certain range of the regularization parameter
regularization_parameters_tv = linspace(5, 20, 20);
PSNR_tv = zeros(size(regularization_parameters_tv));
for idx = 1:numel(regularization_parameters_tv)
    lambda = regularization_parameters_tv(idx);
    I_prox_tv = TV_prox(I_noisy, 'lambda', lambda, 'maxiter', prox_tv_maxiters, 'tol', prox_tv_tol, 'verbose', 0);
    PSNR_tv(idx) = PSNR(I, I_prox_tv);
end

%% Visualizing plots

figure,
fontsize = 12;
subplot(121),
scatter(regularization_parameters_1, PSNR_1, 'filled');
grid on;
grid minor;
xlabel('Lambda');
ylabel('PSNR');
title('L1 prox','fontsize',fontsize,'interpreter','latex');
subplot(122)
scatter(regularization_parameters_tv, PSNR_tv, 'filled');
grid on;
grid minor;
xlabel('Lambda');
ylabel('PSNR');
title('TV prox','fontsize',fontsize,'interpreter','latex');