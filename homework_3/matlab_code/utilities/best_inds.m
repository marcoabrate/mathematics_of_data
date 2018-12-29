function [ mask_best ] = best_inds( f, rate )
kkk = abs((fft2(f)));


[~,ind_best]=sort(kkk(:),'descend');
ind_best = ind_best(1:floor(numel(ind_best)*rate));
mask_best=zeros(size(f));mask_best(ind_best)=1;

end

