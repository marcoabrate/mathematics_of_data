clear all
%% Problem size - image side
%addpath('export_fig');
addpath('utilities/');

load('IMAGES/brain.mat')
load('IMAGES/indices_brain.mat')

m           = 256; 
rate        = 0.20; %
p           = numel(ind); % omega

N           = m^2;
Ne          = 2*N;

f           = im_brain; % real image

f_print = sqrt(real(f).^2+imag(f).^2);

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
y_real = real(measurement_forward(f));
y_imag = imag(measurement_forward(f));
y = [y_real; y_imag];
sz = size(y_real, 1);
sze = 2*sz;

% Optimization parameters
maxit = 100;
tolx  = 1e-5;

%% WAVELET RECOVERY
% Initial point
x0      = [zeros(N, 1); zeros(N, 1)];

fx      = @(x) 0.5 * norm( y_real - real(forward_operator(x(1:N))) + imag(forward_operator(x(N+1:Ne))) , 2)^2 + ...
    0.5 * norm( y_imag - imag(forward_operator(x(1:N))) - real(forward_operator(x(N+1:Ne))) , 2)^2;

gradf = @(x) gradient(y, forward_operator, adjoint_operator, x, sz, N);

%% Solving FISTA on a certain range of the regularization parameter
startValue = 5e-5;
endValue = 5e-4;
nElements = 100;
stepSize = (endValue-startValue)/(nElements-1);
regularization_parameters = startValue:stepSize:endValue;
PSNR_wav = zeros(size(regularization_parameters));

for idx = 1:numel(regularization_parameters)
    regularization_parameter_lasso = regularization_parameters(idx);
    fprintf('lambda = %f\n', regularization_parameter_lasso);
    
    gx = @(x) regularization_parameter_lasso * norm(x(1:N)+1i*x(N+1:Ne), 1);
    proxg = @(x, gamma) proxL1norm_complex(x, regularization_parameter_lasso*gamma);
    % solving FISTA
    f_wav = FISTA(fx, gx, gradf, proxg, x0, Lips, maxit, tolx, 0);
    % setting the warm start for the next iteration
    %x0 = f_wav;
    f_rec = representation_operator_trans(f_wav(1:N) + 1i*f_wav(N+1:Ne));
    f_rec = reshape(f_rec, [m, m]);
    f_power = sqrt(real(f_rec).^2+imag(f_rec).^2);
    PSNR_wav(idx) = PSNR(f_print, f_power);
    fprintf('PSNR = %f\n\n', PSNR_wav(idx));
end

%% VISUALIZING results

figure,
fontsize = 12;
scatter(fliplr(regularization_parameters), fliplr(PSNR_wav), 'filled');
grid on;
grid minor;
xlabel('Regularization parameter');
ylabel('PSNR');
title('FISTA algorithm','fontsize',fontsize,'interpreter','latex');