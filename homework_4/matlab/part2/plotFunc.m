function [  ] = plotFunc( mEst, C, x )
%PLOTFUNC This function will be used in FrankWolfe to plot intermediate results.

    hfig = figure(101);
    hfig.Position = [100,100,800,270];
    subplot(121)
    imshow(x)
    title('original blurred image');
    subplot(122)
    xEst = -C(mEst);
    xEst = xEst - min(xEst(:));
    xEst = xEst./max(xEst(:));
    imshow(xEst);
    title('reconstruction by blind deconvolution')
    drawnow

end

