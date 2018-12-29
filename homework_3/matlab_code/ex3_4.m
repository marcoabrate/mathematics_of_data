clear all
%% Problem size - image side
%addpath('export_fig');
addpath('utilities/');
load('IMAGES/brain.mat')
load('IMAGES/indices_brain.mat')

m           = 256; 
rate        = 0.20;

N           = m^2;
Ne          = 2*N;

f           = im_brain; % real image

p           = floor(rate*N); % number of considered indices

ind_mat     = zeros(m, m);

%% Just set one at a time to ONE, the others to ZERO
random      = 0;
given       = 0;
oracle      = 1;

%% Random indices
if random
    ind = randperm(N, p); % p random indices
    ind_mat(ind) = 1;
end


%% Given indices
if given
    ind_mat(ind) = 1;
end

%% Oracle's indices
if oracle
    ind_mat = best_inds(f, rate);
    ind = find(ind_mat);
end


%% Printing the real image
f_print = sqrt(real(f).^2+imag(f).^2);
figure,
imagesc(f_print), axis image off;
title('Real image','interpreter','latex');

%% Printing the masks
figure,
imagesc(ind_mat), axis image off, colormap gray;
title('Mask','interpreter','latex');


%% Wavelet operators
% Define the function handles that compute
% the products by W (DWT) and W' (inverse DWT)

wav         = daubcqf(8);
level       = log2(m); % Maximum level

WT          = @(x) midwt(real(x),wav,level) + 1i*midwt(imag(x),wav,level)  ; % From wavelet coefficients to image
W           = @(x) mdwt(real(x),wav,level) +  1i*mdwt(imag(x),wav,level)   ;  % From image to wavelet coefficients

 %% Define operators

% Vectorized transformations
representation_operator         = @(x) reshape(W(reshape(x,[m,m])),[N,1]);
representation_operator_trans   = @(x) reshape(WT(reshape(x,[m,m])),[N,1]);

% Measurement operators
measurement_forward             = @(x) fft2fwd_without_fftshift(reshape(x,[m,m]),ind);      % P_Omega F
measurement_backward            = @(x) reshape(fft2adj_without_fftshift(x,ind,m,m),[N,1]);  % F^T P_Omega ^T

% Define the overall operator
forward_operator                = @(x) measurement_forward(representation_operator_trans(x)); % A
adjoint_operator                = @(x) representation_operator(measurement_backward(x)); % A^T

%% Defining parameters
Lips    = sqrt(2);

% Generate measurements on complex domain and transfer to real domain
b = measurement_forward(f);
y_real = real(measurement_forward(f));
y_imag = imag(measurement_forward(f));
y       = [y_real; y_imag];
sz = size(y_real, 1);
sze = 2*sz;

% Optimization parameters
maxit = 100;
tolx  = 1e-5;

%% WAVELET RECOVERY
% regularization parameter
regularization_parameter_lasso  = 3e-4;
% Initial point
x0      = [zeros(N, 1); zeros(N, 1)];

fx      = @(x) 0.5 * norm( y_real - real(forward_operator(x(1:N))) + imag(forward_operator(x(N+1:Ne))) , 2)^2 + ...
    0.5 * norm( y_imag - imag(forward_operator(x(1:N))) - real(forward_operator(x(N+1:Ne))) , 2)^2;
gx      = @(x) regularization_parameter_lasso * norm(x(1:N)+1i*x(N+1:Ne), 1);
proxg   = @(x, gamma) proxL1norm_complex(x, regularization_parameter_lasso*gamma);

gradf = @(x) gradient(y, forward_operator, adjoint_operator, x, sz, N);

% applying FISTA
time_wav = tic;
f_wav = FISTA(fx, gx, gradf, proxg, x0, Lips, maxit, tolx, 1);
time_wav = toc(time_wav);

%% LINEAR DECODER
time_lin = tic;
f_est = measurement_backward(b);
time_lin = toc(time_lin);

%% VISUALIZING results
f_rec = representation_operator_trans(f_wav(1:N) + 1i*f_wav(N+1:Ne));
f_rec = reshape(f_rec, [m, m]);
f_power = sqrt(real(f_rec).^2+imag(f_rec).^2);
PSNR_wav = PSNR(f_print, f_power);

figure,
imagesc(f_power), axis image off;
title(sprintf('Reconstructed image (FISTA, time %f s)\nPSNR = %.2f', time_wav, PSNR_wav),'interpreter','latex');

f_est = reshape(f_est, [m, m]);
f_est_power = sqrt(real(f_est).^2+imag(f_est).^2);
PSNR_est = PSNR(f_print, f_est_power);

figure,
imagesc(f_est_power), axis image off;
title(sprintf('Reconstructed image (linear, time: %f s)\nPSNR = %.2f', time_lin, PSNR_est),'interpreter','latex');