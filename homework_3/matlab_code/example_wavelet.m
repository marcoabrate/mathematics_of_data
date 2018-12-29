addpath('utilities');

%% Read EPFL image
I = imread('IMAGES/epfl_1024.jpg');
I = double(I);
m = size(I,1);

%% WAVELETS OPERATORS 
% Define the function handles that compute
% the products by W (inverse DWT) and W' (DWT)
wav      = daubcqf(8);
level    = log2(m); % Maximum level
W        = @(x) midwt(x,wav,level); % From wavelet coefficients to image
WT       = @(x) mdwt(x,wav,level);  % From image to wavelet coefficients

%% Test on EPFL image
% Wavelet transform
I_wav = WT(I);
% Inverse wavelet transform
I_2 = W(I_wav);

%% Plotting the results
figure,
fontsize = 16;
subplot(131),
imagesc(I), axis image off;
title('Original','fontsize',fontsize,'interpreter','latex');
subplot(132),
imagesc(log(abs(I_wav))), axis image off
title('Wavelet coefficients','fontsize',fontsize,'interpreter','latex');
subplot(133)
imagesc(I_2), axis image off; colormap gray
title('Inverse wavelet transform','fontsize',fontsize,'interpreter','latex');