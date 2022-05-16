function [p,testStat] = permutation_test(sample1,sample2,nvp)
% Calculates distribution of test statistic (by default, difference in
% sample means, but user may pass in a custom test statistic) under n
% permutations and returns p value and test statistic. Operates on vectors
% of scalar observations. If the combined number of observations between
% the two samples is 10 or less, an exact test using all possible
% permutations may be requested. For all tests with more than 10 samples,
% an approximation of the null (using random permutations with the number
% set by user) will be used.
%
% PARAMETERS
% ----------
% sample1 -- Vector of scalar observations.
% sample2 -- Vector of scalar observations.
% Name-Value Pairs (nvp)
%   'statFun'       -- (string | function handle). May have string value of
%                      'sampleMean' or 'sampleMedian' in which case the
%                      test statistic is calculated as the sample mean or
%                      median, resepctively. Otherwise, pass statFun as a
%                      handle to a function accepting two vectors (samples)
%                      as inputs and returning a scalar output (the test
%                      statistic).
%   'exact'         -- Logical true or false, specifying whether to use an
%                      exact test featuring enumeration of all possible
%                      permutations (if true) or to use random permutations
%                      (if false). Note that if the combined number of
%                      observations between the two samples is greater than
%                      10, a true value for this name-value pair will be
%                      overriden and random permutations used instead (with
%                      the number being equal to 'nPermutations').
%   'nPermutations' -- Number of permutations to use to generate null
%                      distribution, if 'exact' is false. Will be ignored
%                      if 'exact' is true AND the total number of
%                      observations between both samples is 10 or less.
%                      Default = 10000.
%   'measure'       -- String value, either 'mean' (default) or 'median',
%                      specifying whether to calculate the test statistic
%                      as the difference in sample means or sample medians.
%   'sidedness'     -- Sidedness of the p value. May have string value,
%                      'two-sided', 'left', or 'right'. Note that the
%                      two-sided p-value calculation here assumes a
%                      symmetric null distribution.
%   'nullMethod'    -- (String: 'absoluteValue' | 'byTail'). If
%                      'absoluteValue', two-sided p values will be computed
%                      using the absolute value; note that this assumes a
%                      symmetric null distribution. If 'byTail', two-sided
%                      p values are computed by doubling the smaller tail
%                      (interpretable as a correction for testing both
%                      tails); note that 'byTail' requires the functions
%                      'tail_prob' and 'check_obs_vs_null'.
%   'plot'          -- Logical true or false, specifying whether or not to
%                      plot histogram of null distribution, with observed
%                      difference in sample means/medians marked.
%   'nBins'         -- Scalar value that is number of bins to be used in
%                      histogram of null distribution.
%   'newFig'        -- Logical true or false. If true, a new figure is
%                      generated for the histogram plot. If false,
%                      generation of a new figure is suppressed, allowing
%                      the histogram to be plotted onto existing axes
%                      (e.g., a subplot).
%   'waitBar'       -- Logical true or false specifying whether or not to
%                      display waitbar during computation of null
%                      distribution. This waitbar is stepped at decile
%                      increments over the permutations rather than with
%                      each permutation, so it generally does not slow
%                      performance very noticeably, regardless of the
%                      number of permutations.
%
% RETURNS
% -------
% p       -- Tail probability of the null distribution. 
% obsDiff -- Observed difference in sample means or medians.
%
% Author: Jonathan Chien. Version 3.0. Last updated 5/7/22. Now allows user
% to pass function handle for computing test statistic. Also streamlined
% the preparation of permutations and added in other options for p value
% computation.

arguments
    sample1
    sample2
    nvp.statFun = 'sampleMean'
    nvp.exact = false
    nvp.nPermutations (1,1) = 10000
    nvp.measure string = 'mean'
    nvp.sidedness string = 'two-tailed'
    nvp.nullMethod = 'absoluteValue'
    nvp.plot = true
    nvp.nBins (1,1) = 50
    nvp.newFig = true
    nvp.waitBar = true
end

% Ensure inputs are column vectors.
if isrow(sample1), sample1 = sample1'; end
if isrow(sample2), sample2 = sample2'; end

% Pool and check samples.
allObs = [sample1; sample2]; nAllObs = length(allObs);
if any(isnan(sample1)), warning('NaNs present in sample 1.'); end
if any(isnan(sample2)), warning('NaNs present in sample 2.'); end


%% Calculate test statistic.

if strcmp(nvp.statFun, 'sampleMean')
    testStat = mean(sample1, 'omitnan') - mean(sample2, 'omitnan');
elseif strcmp(nvp.statFun, 'sampleMedian')
    testStat = median(sample1, 'omitnan') - median(sample2, 'omitnan');
else % Custom statistic passed by user
    assert(isa(nvp.statFun, 'function_handle'), ...
           "If 'statFun' does not have a string value of 'sampleMean' " + ...
           "or 'sampleMedian', pass in a function handle that takes in " + ...
           "two samples as vectors and returns a scalar test statistic.")
    testStat = nvp.statFun(sample1, sample2);
    assert(isscalar(testStat), 'Test statistic must be scalar.')
end


%% Generate null distribution

% Optionally initialize waitbar and calculate 10% increments at which to
% update waitbar (the 9 deciles plus the "10th decile" = 100%).
if nvp.waitBar
    w = waitbar(0, 'Initializing...');
    if nvp.exact && nAllObs <= 10
        deciles = prctile(1:size(perms(1:nAllObs),1), 10:10:100);
    else
        deciles = prctile(1:nvp.nPermutations, 10:10:100);    
    end
    nDeciles = length(deciles);
    iDecile = 1;
else
    disp('Computing null distribution...')
end

% Prepare permutations (exact or random).
if nvp.exact && nAllObs <= 10  
    % Enumerate all possible permutations and preallocate.
    permutations = perms(1:nAllObs);
    nPerms = size(permutations, 1);
    permStat = NaN(nPerms, 1);

else
    % If an exact test was requested for two samples whose combined
    % elements number more than 10, warn the user that this request will be
    % ignored, as the number of permutations quickly becomes intractable.
    if nvp.exact
        warning(['Exact test requested but enumerating all permutations ' ...
                 'of more than about 10 total samples is impractical. ' ...
                 'Random permutations will be used to approximate null ' ...
                 'distribution instead, with nPermutations = 10000 by ' ...
                 'default if unset.'])
    end

    % Generate random permutations and preallocate.
    nPerms = nvp.nPermutations;
    permutations = NaN(nPerms, nAllObs);
    for iPerm = 1:nPerms
        permutations(iPerm,:) = randperm(nAllObs);
    end
    permStat = NaN(1, nPerms);
end
  
% Generate null distribution using prepared permutations.
for iPerm = 1:nPerms
    % Permute sample labels and partition observations.
    permObs = allObs(permutations(iPerm,:));
    permSample1 = permObs(1:length(sample1));
    permSample2 = permObs(length(sample1)+1:end);

    % Calculate test statistic on permuted data. Will not repeat validation
    % steps here as it should have been checked when computing test
    % statistic on original unpermuted data.
    if strcmp(nvp.statFun, 'sampleMean')
        permStat(iPerm) = mean(permSample1, 'omitnan') - mean(permSample2, 'omitnan');
    elseif strcmp(nvp.statFun, 'sampleMedian')
        permStat(iPerm) = median(permSample1, 'omitnan') - median(permSample2, 'omitnan');
    else % Custom statistic passed by user
        permStat(iPerm) = nvp.statFun(permSample1, permSample2);
    end

    % Display progress at decile increments.
    if nvp.waitBar && iPerm > deciles(iDecile)
        waitbar(iDecile/nDeciles, w, 'Computing null distribution...')
        iDecile = iDecile + 1;
    end
end

if nvp.waitBar, close(w); else, disp('Finished.'); end


%% Compute p value

% Calculate p value. Note that the permutations for an exact test include
% the identity permutation, so p values will also not be zero. If using
% random permutations, see Phipson & Smyth 2010 for why using the same
% computation as the exact test results in understatement of the p value by
% about 1/nPermutations (instead we use the slightly more conservative
% estimate).
if strcmp(nvp.nullMethod, 'absoluteValue')
    switch nvp.sidedness
        case 'two-tailed'
            if nvp.exact && nAllObs <= 10
                p = mean(abs(permStat) > abs(testStat));
            else
                p = (sum(abs(permStat) > abs(testStat)) + 1) ...
                    / (nvp.nPermutations + 1);
            end
        case 'left-tailed'
            if nvp.exact && nAllObs <= 10
                p = mean(permStat < testStat);
            else
                p = (sum(permStat < testStat) + 1) / (nvp.nPermutations + 1);
            end
        case 'right-tailed'
            if nvp.exact && nAllObs <= 10
                p = mean(permStat > testStat);
            else
                p = (sum(permStat > testStat) + 1) / (nvp.nPermutations + 1);
            end
    end

% Using my package from other applications e.g. decoding, geometry.
elseif strcmp(nvp.nullMethod, 'byTail') 
    assert(exist('tail_prob.m', 'file') == 2 ...
           && exist('check_obs_vs_null.m', 'file') == 2, ...
           "Functions 'tail_prob.m' and 'check_obs_vs_null' required " + ...
           "for the 'byTail' option.")
    p = tail_prob(testStat, permStat, 'type', nvp.sidedness, 'exact', nvp.exact);

else
    error("Unsupported value for 'nullMethod'.")
end
    

%% Optional plotting

% Plot distribution of test statistic.
if nvp.plot
    if nvp.newFig, figure; end
    hold on
    histogram(permStat, nvp.nBins, ...
              'EdgeColor', 'none', 'FaceColor', [0.3010, 0.7450, 0.9330])
    xlabel(sprintf('Difference in permuted sample %ss', nvp.measure))
    ylabel('Count')
    h = plot([testStat testStat], ylim, '--');
    legend(h, sprintf('Observed difference in sample %ss = %d (p = %d)', ...
                      nvp.measure, testStat, p), ...
           'Location', 'northwest')
end

end
