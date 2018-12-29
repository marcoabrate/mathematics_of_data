addpath('utilities');

m = 512;

%% Set TV prox parameters
prox_tv_maxiters            = 100;
prox_tv_tol                 = 1e-5;
regularization_parameter_tv = 1;

%% Compute TV norm of random image
I = randn(m);
I_tv_norm = TV_norm(I,'iso');

%% Compute proximity operator
time_tv   = tic;
I_prox    = TV_prox(I, 'lambda', regularization_parameter_tv, 'maxiter', prox_tv_maxiters, 'tol', prox_tv_tol, 'verbose', 0);
time_tv   = toc(time_tv);

%% Compute TV norm of proximal image
I_prox_tv_norm = TV_norm(I_prox,'iso');

%% Compare the two norms

[I_tv_norm, I_prox_tv_norm]

%% Visualize images to see the effect of Total Variation

figure,
fontsize = 16;
subplot(121),
imagesc(I,[min(I(:)), max(I(:))]), axis image off, colormap jet
title('Original','fontsize',fontsize,'interpreter','latex');
subplot(122)
imagesc(I_prox,[min(I(:)), max(I(:))]), axis image off
title('Proximal point','fontsize',fontsize,'interpreter','latex');