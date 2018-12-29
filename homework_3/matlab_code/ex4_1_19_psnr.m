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
regularization_parameter_tv  = 5;
% Initial point
x0      = zeros(m, m);

fx      = @(x) 0.5 * norm(y - forward_operator(x), 'fro')^2;
gx      = @(x) regularization_parameter_tv * TV_norm(x, 'iso');
proxg   = @(x, gamma) TV_prox(x, regularization_parameter_tv*gamma);

gradf = @(x) gradient19(y, forward_operator, adjoint_operator, x);

%% Solving FISTA on a certain range of the regularization parameter

regularization_parameters = linspace(3, 7e-1, 20);
PSNR_tv = zeros(size(regularization_parameters));

for idx = 1:numel(regularization_parameters)
    regularization_parameter_tv = regularization_parameters(idx);
    fprintf('lambda = %f\n', regularization_parameter_tv);
    
    gx      = @(x) regularization_parameter_tv * TV_norm(x, 'iso');
    proxg   = @(x, gamma) TV_prox(x, 'lambda', gamma*regularization_parameter_tv, ...
        'maxiter', maxit/5, 'tol', tolx*1e3, 'verbose', 0);

    % solving FISTA
    I_rec = FISTA19(fx, gx, gradf, proxg, x0, Lips, maxit, tolx, 0);
    % setting the warm start for the next iteration
    %x0 = I_rec;
    PSNR_tv(idx) = PSNR(I, I_rec);
    fprintf('PSNR = %f\n\n', PSNR_tv(idx));
end

%% VISUALIZING results

figure,
fontsize = 12;
scatter(fliplr(regularization_parameters), fliplr(PSNR_tv), 'filled');
grid on;
grid minor;
xlabel('Regularization parameter');
ylabel('PSNR');
title('FISTA algorithm (TV norm)','fontsize',fontsize,'interpreter','latex');