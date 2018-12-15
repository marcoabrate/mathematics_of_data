% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % EE556 - MATHMATICS OF DATA  % % % % % % % % % % % % FALL 2017 % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% BLIND DECONVOLUTION BY CONVEX PROGRAMMING % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% PURPOSE: We will solve the following problem formulation with % % % % % %
% Frank-Wolfe's method.                                    
%                   min_x  0.5*norm(A(x) - b)^2        
%                   s.t.:  norm_nuc(x) <= kappa,     
%
% Laboratory for Information and Inference Systems (LIONS)
% Ecole Polytechnique Federale de Lausanne (EPFL) - SWITZERLAND
% Last modification: November 23, 2017
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

%% Clear varibles, add dependencies, etc.
clearvars;
addpath('PROPACK');

%% Mappings between n dimensional complex space and 2n dimensional real space
real2comp = @(x) x(1:size(x,1)/2) + 1i*x(size(x,1)/2+1:end);
comp2real = @(x) [real(x); imag(x)];

%% Load Data
x = double(rgb2gray(imread('blurredplate.jpg')));
x = x(61:188,41:296);
x = x./max(abs(x(:)));
figure(1)
imshow(x)
%x = x/norm(x,'fro');
imsize1 = size(x,1); 
imsize2 = size(x,2); 
imsize = imsize1*imsize2; 

%% Reshaping operators matrix to vector and vector to matrix
mat = @(x) reshape(x,imsize1,imsize2);
vec = @(x) x(:);

%% Set the measurement vector b
b = comp2real(vec(fft2(fftshift(x))));

%% Roughly estimate the support the blur kernel
K1 = 17;
K2 = 17;
Indw = zeros(imsize1,imsize2);
Indw(imsize1/2-(K1+1)/2+2:imsize1/2+(K1+1)/2,imsize2/2-(K2+1)/2+2:imsize2/2+(K2+1)/2) = 1;
% above, for simplicity we assume K1 and K2 odd,
% if they are not 1 pixel probably won't cause much trouble
figure(2)
imshow(reshape(ones - Indw,[imsize1, imsize2]))
title('estimated support of blur kernel');
Indw = Indw(:);
kernelsize = nnz(Indw);
[Indi,Indv] = find(Indw);

%% Define operators Bop and Cop
Bmat = sparse(Indi, 1:kernelsize, Indv, imsize, kernelsize );
Bop  = @(x) mat(Bmat*x);
BTop = @(x) Bmat'*vec(x);

%% Compute and display wavelet coefficients of the original and blurred image
[~,l] = wavedec2(x,4,'db1');
Cop  = @(x) waverec2(x,l,'db1');
CTop = @(x) wavedec2(x,4,'db1')';

%% Define operators
% We provide three operators:
% 1- Aoper(m,n,h) is equal to A*(m*n*h'). In other words, Aoper applies linear operator A
% to the rank 1 matrix (m*n*h'). Implmenting a special instance of A, we gain storage
% and computational efficiencies. Note that in Frank-Wolfe, we need operator A to 
% compute the gradient A'*(AX - b). Note that X in general is not rank 1, but we can 
% compute AX using Aoper because the updates on X are rank one. Using linearity, we can
% keep track of AX directly. See the function Frank-Wolfe for more details. 
% 2- AToper.matvec(y,w) computes (A'*y)*x, without generating A'*y in the ambient 
% dimensions. By doing this, we gain storage and computational efficiencies. Note that we
% need A' to compute the gradient A'*(AX-b). Be careful here, that you do not need to form
% the gradient A'*(AX-b), and in fact you should avoid it. We only need to run lmo with 
% the gradient. We can use the power method or the Lanczos method (eigs) to compute 
% top singular vectors. Both of these methods relies on multiplying A'*(AX-b) by a random 
% vector again and again. Hence, we only need an operator that computes (A'*(AX-b))*w for
% a given x. Hence, we actually use AToper.matvec(AX-b,w) to solve lmo. 
% 3- Similar to AToper.matvec(y,w), but this time we compute w'*(A'*y), and will be used 
% in the lmo. 
Aoper = @(m, n, h) comp2real(1/sqrt(imsize)*n*vec(fft2(Cop(m)).*fft2(Bop(h))));
AToper.matvec  = @(y, w) CTop(real(fft2(mat(conj(real2comp(y)).*vec(fft2(Bop(w)))))))/sqrt(size(y,1)/2);
AToper.rmatvec = @(y, w) BTop(real(ifft2(mat(real2comp(y).*vec(ifft2(Cop(w)))))))*(size(y,1)/2)^1.5;

%% Run Frank-Wolfe's method
MaxIters = 200;
kappa = 1000;
[xFW, outFw] = FrankWolfe( Aoper, AToper, b, kernelsize, imsize, kappa, MaxIters, @(m)plotFunc(m,Cop,x));

%% NOTE: This experiment is based on the theory and the codes publised in
% 'Blind Deconvolution using Convex Programming' by A.Ahmed, B.Recht and
% J.Romberg.
