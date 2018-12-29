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
ind = randi(N, p, 1); % p random indices
ind_mat(ind) = 1;

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
measurement_backward            = @(x) ind_mat'.*x;  % P_Omega ^T

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
% Initial point
x0      = zeros(m, m);

fx      = @(x) 0.5 * norm(y - forward_operator(x), 'fro')^2;

gradf = @(x) gradient18(y, forward_operator, adjoint_operator, x);

%% Solving FISTA on a certain range of the regularization parameter

regularization_parameters = linspace(10, 1, 30);
PSNR_1 = zeros(size(regularization_parameters));

for idx = 1:numel(regularization_parameters)
    regularization_parameter_lasso = regularization_parameters(idx);
    fprintf('lambda = %f\n', regularization_parameter_lasso);
    
    gx      = @(x) regularization_parameter_lasso * norm(reshape(x, [N,1]), 1);
    proxg   = @(x, gamma) proxL1norm(x, regularization_parameter_lasso*gamma);

    % solving FISTA
    I_wav = FISTA18(fx, gx, gradf, proxg, x0, Lips, maxit, tolx, 0);
    % setting the warm start for the next iteration
    %x0 = I_wav;
    I_rec = representation_operator_trans(I_wav);
    PSNR_1(idx) = PSNR(I, I_rec);
    fprintf('PSNR = %f\n\n', PSNR_1(idx));
end

%% VISUALIZING results

figure,
fontsize = 12;
scatter(fliplr(regularization_parameters), fliplr(PSNR_1), 'filled');
grid on;
grid minor;
xlabel('Regularization parameter');
ylabel('PSNR');
title('FISTA algorithm (L1 norm)','fontsize',fontsize,'interpreter','latex');