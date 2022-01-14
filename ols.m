function stats = ols(X,Y,nvp)
% Fit an ordinary least squares model for predictor/independent variable(s)
% in X and response/dependent variable(s) in Y. X is an n x p design
% matrix, and Y is an n x m matrix of response variables. If (not counting
% a possible intercept term) p = m = 1 , this is univariate simple
% regression. If m = 1, and p > 1, this is univariate multiple regression.
% If both m > 1 and p > 1, this is multivariate multiple regression. For
% all cases, the OLS solution consists of a p x m matrix of beta
% coefficients, whose i_th j_th element is the weight of the i_th predictor
% toward the j_th response variable. Note that the design matrix, X, may or
% may not feature a column of ones to allow for an intercept term; if it
% does feature a column of ones, however, this column must be the final
% column of X. Optionally, the user may request the computation of
% coefficients of partial determination and p values for each predictor,
% for each response variable. Note, however, that the method employed to
% calculate p-values here is correct for hand calculations but likely not
% as stable or efficient as built-in implementations. The advantage of this
% function is a no-frills vectorized version of OLS over multiple response
% variables (which may also be useful when fitting models over bins, where
% each bin is a response variable), especially where p values are not as
% important.
%
% PARAMETERS
% ----------
% X -- n x p design matrix featuring n observations and p predictors. If
%      p = 1 (not counting a possible intercept term), this is simple
%      regression. X may or may not feature a column of ones to allow for
%      an intercept term, but if it does, this column must be the last
%      column of X.
% Y -- n x m matrix of response variables, where the i_th j_th element is
%      the response of the m_th dependent variable on the i_th
%      observation/trial. If m = 1, this is univariate regression.
% Name-Value Pairs
%   'cpd'  -- (1|0, default = 1). Specify whether or not to compute
%             coefficients of partial determination for each predictor over
%             all response variables.
%   'pVal' -- (1|0, default = 0). Specify whether or not to compute
%             p-values for beta coefficients.
% 
% RETURNS
% -------
% stats -- Scalar struct with regrssion results in the following fields:
%   .beta      -- p x m matrix of beta coefficients, whose i_th j_th
%                 element is the weight of the i_th predictor toward the
%                 j_th response variable. Note that p here does not count
%                 an intercept term, if one was included; i.e., the
%                 function will find an OLS solution but will remove the
%                 row corresponding to weights on the intercept term in the
%                 returned array here.
%   .predicted -- n x m matrix whose j_th column is the projection of the
%                 j_th column of Y into the column space of X. The
%                 elemeents of this vector are the model's predicted values
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
%                 variable and its predicted value under the model, i.e.,
%                 the square of the cosine of the angle between the
%                 unit-length j_th column of Y and its unit-length
%                 projection onto the column space of X).
%   .cpd       -- Optional p x m matrix of coefficients of partial
%                 determination, where the i_th j_th element is the
%                 coefficient for the i_th predictor on the j_th response
%                 variable (again, there is no coefficient here for the
%                 intercept term, if one was included). If 'cpd' = false
%                 (see PARAMETERS), this field will be absent from the
%                 returned stats struct.
%   .p         -- Optional p x m matrix of p values, where the i_th j_th
%                 element is p value attaching to the beta coefficient of
%                 the i_th predictor toward the j_th response variable
%                 (i.e., the i_th j_th element of the betas matrix (see
%                 above)). If 'pVal' = false (see PARAMETERS), this field
%                 will be absent from the returned stats struct.
%   .vif       -- p x 1 vector whose i_th element is the variance inflation
%                 factor (VIF) for the i_th predictor (again, no value for
%                 the intercept term is returned, if an intercept term was
%                 included).
%
% Author: Jonathan Chien 1/12/22. Last edit: 1/13/22.


arguments
    X
    Y
    nvp.cpd = true
    nvp.pVal = false
end

% Obtain/check array sizes/values.
nObs = size(X, 1);
nDv = size(Y, 2); % number of dependent vars
nIv = size(X, 2) - 1; % number of independent vars, excluding constant term
assert(nObs == size(Y, 1))
if ~any(sum(X) == nObs)
    warning('X was passed in without a column of ones. The fitted model will have no intercept.')
elseif any(sum(X(:,1:end-1)) == nObs)
    error('If X includes a vector of ones, it must be placed in the final column of X')
end

% Fit OLS model across all response variables.
stats.beta = X \ Y; 
stats.predicted = X * stats.beta;
stats.resid = Y - stats.predicted;
stats.ssr = sum( (stats.resid).^2 );

% Calculate coefficient of determination for all dependent variables. 
stats.sst = sum( (Y - mean(Y)).^2 );
stats.cd = 1 - (stats.ssr ./ stats.sst);

% Calculate coefficient of partial determination for each predictor, for
% each dependent variable.
if nvp.cpd
    stats.cpd = NaN(nIv, nDv);
    for iIv = 1:nIv 
        XRed = X(:,setdiff(1:nIv+1,iIv));
        statsRed.beta = XRed \ Y;
        statsRed.predicted = XRed * statsRed.beta;
        statsRed.resid = Y - statsRed.predicted;
        statsRed.ssr = sum( (statsRed.resid).^2 );
        stats.cpd(iIv,:) = (statsRed.ssr - stats.ssr ) ./ statsRed.ssr;
    end
end

% If a column of ones is present, remove it and its beta coefficient once
% calculations are complete.
if sum(X(:,end)) == nObs
    stats.beta(end,:) = [];
    X(:,end) = [];
end

% Optionally attach p values to beta cofficients. These are "hand
% calculations" and likely will not match the numerical efficiency and
% stability of built-in implementations, for applications where this
% really matters (usually it should be fine).
if nvp.pVal
    dof = nObs - nIv - 1; 
    sigmas = stats.ssr / dof;
    Xmu = X - mean(X);
    se = sqrt(diag(inv(Xmu'*Xmu)) .* sigmas); % .* is outer product in multivariate case
    t = stats.beta ./ se;
    stats.p = 2 * tcdf(abs(t), dof, 'upper');
end

%  Calculate VIF for each of the predictors.
stats.vif = diag(inv(corrcoef(X)));

end
