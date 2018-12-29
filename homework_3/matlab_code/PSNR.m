% FUNCTION: pout = PSNR(I, I_rec)
% PURPOSE:  Peak Signal-to-Noise Ratio.
%
function pout = PSNR(I, I_rec)
    
    sqrtMSE = (1/size(I,1)) * norm(I - I_rec, 'fro');
    pout = 20*log10(max(I(:))/sqrtMSE);

end
