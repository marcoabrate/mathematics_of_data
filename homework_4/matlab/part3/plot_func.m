output = outHCGM;
font_size = 18;
figure('Position',[100,500,760,270]);
subplot(1,2,1)
loglog(output.iteration, output.feasibility1, 'b'), grid on;axis tight
ylabel('$\|X1-\mathbf{1}\|/\|\mathbf{1}\|$', 'Interpreter', 'latex', 'FontSize', font_size);
xlabel('iters', 'Interpreter', 'latex', 'FontSize', font_size);

subplot(1,2,2)
loglog(output.iteration, output.feasibility2, 'r'), grid on;axis tight
ylabel('dist$(X, \mathbf{R}^{n}_+)$', 'Interpreter', 'latex', 'FontSize', font_size)
xlabel('iters', 'Interpreter', 'latex', 'FontSize', font_size);

figure;
loglog(output.iteration, abs(output.objective-opt_val)/abs(opt_val)), grid on;
title('Relative Objective Residual','Interpreter', 'latex', 'FontSize', font_size);
xlabel('iters', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('$\frac{ |f(X)-f^\star| }{ |f^\star| }$', 'Interpreter', 'latex', 'FontSize', font_size);  

figure;imagesc(xHCGM);axis off;
title('SDP Solution','Interpreter', 'latex', 'FontSize', font_size);