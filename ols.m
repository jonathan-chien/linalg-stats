function stats = ols(X,Y,nvp)
% Fit an ordinary least squares model for predictor/independent variable(s)
% in X and response/dependent variable(s) in Y. X is an n x p design
% matrix, and Y is an n x m matrix of response variables. If (not counting
% a possible intercept term) p = m = 1 , this is univariate simple
% regression. If m = 1, and p > 1, this is univariate multiple regression.
% If both m > 1 and p > 1, this is multivariate multiple regression. For
% all cases, the OLS solution consists of a p x m matrix of beta
% coefficients, whose i_th j_th element is the weight of the i_th predictor
% toward the j_th response variable. 
%
% PARAMETERS
% ----------
% X -- n x p design matrix featuring n observations and p predictors. If
%      p = 1 (not counting a possible intercept term), this is simple
%      regression. X should not feature a column of ones to account for
%      an intercept term; instead see the 'intercept' name-value pair.
% Y -- n x m matrix of response variables, where the i_th j_th element is
%      the response of the m_th dependent variable on the i_th
%      observation/trial. If m = 1, this is univariate regression.
% Name-Value Pairs
%   'intercept' -- (1 (default) | 0). Specify whether or not to append a
%                  column of ones to the design matrix to allow for a
%                  nonzero intercept in the model. Note that if so, this
%                  column (and any associated p values or CPDs) comes last.
%   'cpd'       -- (1 (default) | 0 | 'includeIntercept'). If true,
%                  coefficients of partial determination (CPD) will be
%                  computed for each predictor, not including a possible
%                  intercept term, over all response variables. If
%                  'includeIntercept' CPDs will be calculated for the
%                  intercept as well. If false, computation of CPDs will be
%                  suppressed.
%   'pval'      -- (1|0, default = 1). Specify whether or not to compute
%                  p-values for beta coefficients. 
%   'vif'       -- (1|0, default = 1). Specify whether or not to compute
%                  the variance inflation factors for each of the
%                  predictors (not including any potential intercept).
% 
% RETURNS
% -------
% stats -- Scalar struct with regrssion results in the following fields:
%   .beta      -- p x m matrix of beta coefficients, whose i_th j_th
%                 element is the weight of the i_th predictor toward the
%                 j_th response variable. Note that if a nonzero intercept
%                 term was included, the last element of p corresponds to
%                 this term.
%   .predicted -- n x m matrix whose j_th column is the projection of the
%                 j_th column of Y into the column space of X. The
%                 elements of this vector are the model's predicted values
%                 for the j_th response variable.
%   .resid     -- n x m matrix whose j_th column is the error vector from
%                 the projection of the j_th column of Y into the column
%                 space of X. The elements of the j_th column are the
%                 residuals for the j_th response variable.
%   .ssr       -- 1 x m vector of sum of squared residuals. The j_th
%                 element is the sum of the squared elements of the j_th
%                 column of .resid.
%   .sst       -- 1 x m vector of total sum of squares. The j_th element is
%                 the sum of the squared components of the j_th column of Y
%                 after centering the mean at 0.
%   .cd        -- 1 x m vector whose j_th element is the coefficient of
%                 determination of the model for the j_th response variable
%                 (the square of the correlation between the j_th response
%                 variable and its predicted value under the model; also,
%                 the square of the cosine similarity of the j_th column of
%                 Y and its projection onto the column space of X).
%   .cpd       -- Optional p x m matrix of coefficients of partial
%                 determination, where the i_th j_th element is the
%                 coefficient for the i_th predictor toward the j_th
%                 response variable (there is no coefficient here for the
%                 intercept term by default, but the user can request one
%                 by setting 'cpd' to 'includeIntercept' (see PARAMETERS),
%                 in which case the last row will contain CPDs for the
%                 intercept, with the j_th element corresponding to the
%                 j_th response variable). If 'cpd' = false, this field
%                 will be absent from the returned stats struct.
%   .p         -- Optional p x m matrix of p values, where the i_th j_th
%                 element is the p value attaching to the beta coefficient
%                 of the i_th predictor toward the j_th response variable
%                 (i.e., the i_th j_th element of the betas matrix (see
%                 above)). If 'pval' = false (see PARAMETERS), this field
%                 will be absent from the returned stats struct.
%   .vif       -- p x 1 vector whose i_th element is the variance inflation
%                 factor (VIF) for the i_th predictor (no value for the
%                 intercept term is returned, if one was included, as
%                 calculation of VIFs here requires the correlation matrix
%                 of X). If 'vif' = false (see PARAMETERS), this field will
%                 be absent from the returned struct.
%
% Author: Jonathan Chien 1/12/22. Last edit: 2/4/22.

arguments
    X
    Y
    nvp.intercept = true
    nvp.cpd = true
    nvp.pval = true
    nvp.vif = true
end

% Check array sizes and get number of depedent variables.
nObs = size(X, 1);
assert(nObs == size(Y, 1), 'First array dim sizes of X and Y must be equal.')
nDv = size(Y, 2); 

% Optionally add vector of ones for intercept term. Get number of columns
% of X after possibly adding ones; define nIv (number of independent vars
% as number of columns of X not including intercept, if one was included).
assert(~any(sum(X) == nObs), ...
       "Do not pass in X with a column of ones. If an intercept term is " + ...
       "desired, set 'intercept' to true.")
if nvp.intercept, X = [X ones(nObs, 1)]; end; nCols = size(X, 2); 
if nvp.intercept, nIv = nCols - 1; else, nIv = nCols; end 

% Fit OLS model.
stats.beta = X \ Y; 
stats.predicted = X * stats.beta;
stats.resid = Y - stats.predicted;
stats.ssr = sum( (stats.resid).^2 );

% Calculate coefficient of determination for all dependent variables. 
stats.sst = sum( (Y - mean(Y)).^2 );
stats.cd = 1 - (stats.ssr ./ stats.sst);

% Calculate coefficient of partial determination for each predictor (not
% including intercept by default), for each dependent variable.
if nvp.cpd
    if strcmp(nvp.cpd, 'includeIntercept')
        k = nCols;
        if nCols == nIv
            warning(['A CPD was requested for the intercept term, but ' ...
                     'there is no intercept.'])
        end
    else 
        k = nIv;
    end
    stats.cpd = NaN(k, nDv);
    for iIv = 1:k
        XRed = X(:,setdiff(1:nCols,iIv));
        statsRed.beta = XRed \ Y;
        statsRed.predicted = XRed * statsRed.beta;
        statsRed.resid = Y - statsRed.predicted;
        statsRed.ssr = sum( (statsRed.resid).^2 );
        stats.cpd(iIv,:) = (statsRed.ssr - stats.ssr ) ./ statsRed.ssr;
    end
end

% Optionally attach p values to beta cofficients. 
if nvp.pval
    dof = nObs - nCols; 
    sigmas = stats.ssr / dof;
    se = sqrt(diag(inv(X'*X)) .* sigmas); % .* is outer product in multivariate case
    t = stats.beta ./ se;
    stats.p = 2 * tcdf(abs(t), dof, 'upper');
end

%  Calculate VIF for each of the predictors.
if nvp.vif, stats.vif = diag(inv(corrcoef(X(:,1:nIv)))); end

end
