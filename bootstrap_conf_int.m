function confInt = bootstrap_conf_int(samples,confLevel,nvp)
% For a given sample provided as a vector input, bootstrap a confidence
% interval around the sample mean at the specified confidence level. 
%
% PARMETERS
% ---------
% samples   -- Vector whose elements are i.i.d. random variables.
% confLevel -- Positive integer from 0 to 100 specifying the desired
%              confidence level of the returned interval.
% Name-Value Pairs (nvp)
%   'nBootstraps'  -- Scalar number of bootstrap resamples to draw. Default
%                     = 10000.
%
% RETURNS
% -------
% confInt -- Vector whose first and second elements are, respectively,
%            the lower and upper bounds of the bootstrapped confidence
%            interval.
%
% Author: Jonathan Chien. 2/3/22.

arguments
    samples
    confLevel
    nvp.nBootstraps = 10000
end

% Ensure user passed in a vector of samples and requested a valid
% confidence level.
assert(isvector(samples))
assert(isscalar(confLevel) && confLevel <= 100 && confLevel >= 0, ...
       'confLevel must be a scalar between 0 and 100 inclusive.')

% Calculate sample mean and size; preallocate.
xbar = mean(samples);
sampleSize = length(samples);
delta = NaN(nvp.nBootstraps, 1);

% Draw samples from the resampling distribution and use the variability of
% their means with respect to the sample mean to estimate the the
% variability of the provided sample mean around the true population mean.
for iBoot = 1:nvp.nBootstraps
    delta(iBoot) ...
        = mean(datasample(samples, sampleSize, 'Replace', true)) - xbar;
end

critVals = prctile(delta, [50+confLevel/2 50-confLevel/2]);
confInt = xbar - critVals;

end
