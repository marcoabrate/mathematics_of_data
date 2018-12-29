function y = fft2fwd_without_fftshift(x, ind)


y = (fft2((x)));
y = y(ind);

