function L = power_method(B)

x = randn(size(B,2),1);

for i = 1:25
   x = x / norm(x);
   x = B'*(B*x);
end

L = norm(x);
