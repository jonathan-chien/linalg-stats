function [p,obsDiff] = permutation_test(sample1, sample2, nvp)
% Calculates distribution of test statistic (difference in sample means of
% medians) under n permutations and returns p value and effect size (the
% observed difference in sample means or medians). Operates on scalar
% observations. If the combined number of observations between the two
% samples is 10 or less, an exact test using all possible permutations may
% be requested. For all tests with more than 10 samples, an approximation
% of the null (using random permutations with the number set by user) will
% be used.
%
% PARAMETERS
% ----------
% sample1 -- Vector of scalar observations.
% sample2 -- Vector of scalar observations.
% Name-Value Pairs (nvp)
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
%   'measure'       -- String value, either 'mean' (default) or 'median',
%                      specifying whether to calculate the test statistic
%                      as the difference in sample means or sample medians.
%   'sidedness'     -- Sidedness of the p value. May have string value,
%                      'two-sided', 'left', or 'right'. Note that the
%                      two-sided p-value calculation here assumes a
%                      symmetric null distribution.
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
% Author: Jonathan Chien. Version 2.0. 2/18/21.


arguments
    sample1
    sample2
    nvp.exact = false
    nvp.nPermutations (1,1) = 10000
    nvp.measure string = 'mean'
    nvp.sidedness string = 'two-sided'
    nvp.plot = true
    nvp.nBins (1,1) = 50
    nvp.newFig = true
    nvp.waitBar = true
end

% Ensure inputs are column vectors.
if isrow(sample1), sample1 = sample1'; end
if isrow(sample2), sample2 = sample2'; end

% Pool samples.
allObs = [sample1; sample2];

% Check whether observations contain NaN values.
if any(isnan(sample1)), warning('NaNs present in sample 1.'); end
if any(isnan(sample2)), warning('NaNs present in sample 2.'); end

% Calculate observed difference in sample means/medians.
switch nvp.measure
    case 'mean'
        obsDiff = mean(sample1, 'omitnan') - mean(sample2, 'omitnan');
    case 'median'
        obsDiff = median(sample1, 'omitnan') - median(sample2, 'omitnan');
end
    
% Optionally initialize waitbar and calculate 10% increments at which to
% update waitbar (the 9 deciles plus the "10th decile = 100%).
if nvp.waitBar
    w = waitbar(0, 'Conducting permutation test...');
    if nvp.exact && length(allObs) <= 10
        deciles = prctile(1:size(perms(1:length(allObs)),1), 10:10:100);
    else
        deciles = prctile(1:nvp.nPermutations, 10:10:100);    
    end
    nDeciles = length(deciles);
    iDecile = 1;
else
    disp('Computing null distribution...')
end

% Calculate difference of sample means based on permuted sample labels.
if nvp.exact && length(allObs) <= 10
    
    % Enumerate all possible permutations.
    exactPerms = perms(1:length(allObs));
    nExactPerms = size(exactPerms, 1);
    randDiff = NaN(nExactPerms, 1);
    
    for iPerm = 1:nExactPerms
        % Permute sample labels and partition observations.
        permObs = allObs(exactPerms(iPerm,:));
        permSample1 = permObs(1:length(sample1));
        permSample2 = permObs(length(sample1)+1:end);

        % Calculate difference in means/medians.
        switch nvp.measure
            case 'mean'
                randDiff(iPerm) = mean(permSample1, 'omitnan') ...
                                  - mean(permSample2, 'omitnan');
            case 'median'
                randDiff(iPerm) = median(permSample1, 'omitnan') ...
                                  - median(permSample2, 'omitnan');
        end

        % Display progress at decile increments.
        if nvp.waitBar && iPerm > deciles(iDecile)
            waitbar(iDecile/nDeciles, w, 'Computing null distribution...')
            iDecile = iDecile + 1;
        end
    end
    
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
    
    % Generate approximation of null distribution using random permutations.
    randDiff = NaN(1, nvp.nPermutations);
    for iPerm = 1:nvp.nPermutations
        % Permute sample labels and partition observations.
        permObs = allObs(randperm(length(allObs)));
        permSample1 = permObs(1:length(sample1));
        permSample2 = permObs(length(sample1)+1:end);

        % Calculate difference in means/medians.
        switch nvp.measure
            case 'mean'
                randDiff(iPerm) = mean(permSample1, 'omitnan') ...
                                  - mean(permSample2, 'omitnan');
            case 'median'
                randDiff(iPerm) = median(permSample1, 'omitnan') ...
                                  - median(permSample2, 'omitnan');
        end

        % Display progress at 10% increments.
        if nvp.waitBar && iPerm > deciles(iDecile)
            waitbar(iDecile/nDeciles, w, 'Computing null distribution...')
            iDecile = iDecile + 1;
        end
    end
end

if nvp.waitBar
    close(w)
else
    disp('Finished.')
end

% Calculate p value. Note that the permutations for an exact test include
% the identity permutation, so p values will also not be zero. If using
% random permutations, see Phipson & Smyth 2010 for why using the same
% computation as the exact test results in understatement of the p value by
% about 1/nPermutations (instead we use the slightly more conservative
% estimate).
switch nvp.sidedness
    case 'two-sided'
        if nvp.exact && length(allObs) <= 10
            p = mean(abs(randDiff) > abs(obsDiff));
        else
            p = (sum(abs(randDiff) > abs(obsDiff)) + 1) ...
                / (nvp.nPermutations + 1);
        end
    case 'left'
        if nvp.exact && length(allObs) <= 10
            p = mean(randDiff < obsDiff);
        else
            p = (sum(randDiff < obsDiff) + 1) / (nvp.nPermutations + 1);
        end
    case 'right'
        if nvp.exact && length(allObs) <= 10
            p = mean(randDiff > obsDiff);
        else
            p = (sum(randDiff > obsDiff) + 1) / (nvp.nPermutations + 1);
        end
end
    

% Option to plot distribution of test statistic (difference in sample
% means).
if nvp.plot
    if nvp.newFig, figure; end
    hold on
    histogram(randDiff, nvp.nBins, ...
              'EdgeColor', 'none', 'FaceColor', [0.3010, 0.7450, 0.9330])
    xlabel(sprintf('Difference in permuted sample %ss', nvp.measure))
    ylabel('Count')
    h = plot([obsDiff obsDiff], ylim, '--');
    legend(h, sprintf('Observed difference in sample %ss = %d (p = %d)', ...
                      nvp.measure, obsDiff, p), ...
           'Location', 'northwest')
end

end
