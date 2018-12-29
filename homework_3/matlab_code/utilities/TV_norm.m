function tv_out = TV_norm(X, opts)

% tv_out = TV_norm(X, opts)
% Computes the TV-norm of image X
% opts = 'iso' for isotropic
% opt = 'l1' for anisotropic

[m, n] = size(X);
P{1}   = X(1:m-1, :) - X(2:m, :);
P{2}   = X(:, 1:n-1) - X(:, 2:n);

if strcmpi(opts, 'iso')
    D           = zeros(m, n);
    D(1:m-1, :) = P{1}.^2;
    D(:, 1:n-1) = D(:, 1:n-1) + P{2}.^2;
    tv_out      = sum(sum(sqrt(D)));
else
    tv_out      = sum(sum(abs(P{1}))) + sum(sum(abs(P{2})));
end