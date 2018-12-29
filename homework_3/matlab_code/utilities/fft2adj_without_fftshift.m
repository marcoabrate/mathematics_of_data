function y = fft2adj(x, ind, m,n)

y = zeros([m n]);
y(ind) = x;
y = (ifft2((y)));
y=y(:);
