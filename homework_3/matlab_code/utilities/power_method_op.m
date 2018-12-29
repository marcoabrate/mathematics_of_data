function L = power_method_op(A, AT, N)

x = randn(N,1);

for i = 1:25
   x = x / norm(x);
   x = AT(A(x));
end

L = norm(x);
