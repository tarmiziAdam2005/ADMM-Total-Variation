function sigma = BSNR2WGNsigma(g, BSNR)
%% Compute the standard deviation of the blur image.
%%

sigma = sqrt(norm(g(:)-mean(g(:)),2)^2 /(numel(g)*10^(BSNR/10)));