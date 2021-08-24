function Xhat = dimension_removal(sourcePops,B_AllReps)
% Takes as input predDims, a p x m x r array, where p is the number of
% neurons in source population (V1), m the number of predictive dimensions
% (equal to i, for i = 1:length(nDimsUsedForPrediction)), and r the number
% of repetitions for each dataset. p and r are constant for all cells,
% while m varies with and is equal to the cell index. Each p x m is a
% matrix, B_, whose columns are predictive dimensions from
% ReducedRankRegress, where B_ = B_ols * V, with B = the OLS solution to Y
% = XB (where Y = population activity of target, and X = population
% activity of source) and V the eigenvectors of Y_hat = X * B_ols; of
% course if ridge regression is used, B_ols is replaced by the regularized
% beta weights etc.). Our goal is to find a basis for a subspace orthogonal
% to every vector in B_ with respect to the source activity. To accomplish
% this, we first postmultiply B_ by the covariance matrix of X (recalling
% that the covariance matrix as a linear transformation stretches a unit
% hypersphere into the shape of (the row vectors of) X, i.e. it stretches
% the data in the direction of its eigenvectors (the principal axes) by a
% scale factor equivalent to the associated eigenvalues--this idea of
% pulling the regression dimensions in the direction of the principal axes
% sounds a bit reminiscent of Partial Least Squares). For, M = B_ * Cov(X),
% we then perform the SVD of X. We next note that we generally test a
% number of predictive dimensions that is much smaller than the total
% number of units. As such, B_' is an underdetermined matrix, as is M (for
% B_ = p x m, and Cov = p x p, we have M = m x p, with m << p). Thus, the
% trailing p - m right singular vectors have an associated singular value
% of 0. These right singular vectors are eigenvectors of Cov(X) with
% associated eigenvalue 0 and under a PCA model interpretation correspond
% to principal axes along which there is 0 variance (since the
% dimensionality of the row space is smaller than that of its ambient
% space). More specifically, the row space of M is spanned by the leading m
% right singular vectors, guaranteeing that the row space is orthogonal to
% the subspace spanned by the last p - m right singular vectors (due to the
% orthogonality of all right singular vectors). By choosing these trailing
% singular vectors as a basis, we then have an orthonormal basis for a
% subsapce orthogonal to the row space of M.
%
% PARAMETERS
% ----------
% sourcePops -- nObs x nUnits x nReps array of firing rates/residuals. This
%               corresponds to one dataset from one session. (When called
%               from the dimension_removal_pipeline, this array also
%               corresponds to one value for number of predictive
%               dimensions.)
% B_AllReps  -- nUnits x nPredDims x nReps array of predictive dimensions
%               corresponding to one dataset from one session. (Again, when
%               called from the dimension_removal_pipeline, this array also
%               corresponds to one value for number of predictive
%               dimensions.) This array is best understood as a series of
%               nUnits x nPredDims slices (B_ for one repetition) stored
%               along the third axis (nReps).
%
% RETURNS
% -------
% Xhat -- nObs x (nUnits-nPredDims) x nReps array of firing
%         rates/residuals. This is the same as the input sourcePops,
%         except that population activity along the predictive dimeensions
%         has been removed.
%
% Author: Jonathan Chien Version 1.1. 8/6/21. Last edit: 8/9/21
%   Based on Methods from "Cortical Areas Interact through a Communication
%   Subspace", Semedo et al 2019, Neuron.
%   Requires MATLAB 2020b or later, for the use of the pagemtimes function.


% Determine number of observations, neurons, and repetitions of dataset.
[nObs, nUnits, nReps] = size(sourcePops);

% Calculate covariance matrix of source activity (correlation matrix if
% z-scoring was applied) for each repetition of dataset.
C = pagemtimes(permute(sourcePops - mean(sourcePops), [2 1 3]), ...
               sourcePops - mean(sourcePops)) ...
    / (nObs - 1);

% Obtain M, which is current p x m x r predDims array, right-multiplied
% slicewise by array with corresponding covariance/correlation matrices as
% slices (i.e., C).
M = pagemtimes(permute(B_AllReps,[2 1 3]), C);

% For each repetition of the current dataset, find basis for subspace
% uncorrelated to predictive dimensions wrt source population activity.
nPredDims = size(M, 1);
Q = NaN(nUnits, (nUnits-nPredDims), nReps); 
for iRep = 1:nReps
   [~,S,V] = svd(M(:,:,iRep));
   Q(:,:,iRep) = V(:, nPredDims+1 : end);
   sigmas = diag(S);
   assert(all(ismembertol(sigmas(nPredDims+1:end),0,1e-12)), ...
          'Double-check inputs.')
end

% Change of basis, adopting trailing right singular vectors (with
% associated singular value = 0) as basis for our data observations
% (across repetition). For observations as columns of V1, we would have
% Q' * V1, so to place observations in rows we take (Q' * V1)' = V1' *
% Q, where V1' has observations in rows.
Xhat = pagemtimes(sourcePops, Q);

end