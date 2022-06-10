function out = pca_svd(X,nv)
% Takes as input an m x n matrix, where m = number of observations and n =
% number of variables, and performs PCA via economy-sized SVD, such that
% the rank of the reconstructed input matrix and its reconstructed
% covariance/correlation matrix are reduced to d, where d is the number of
% principal components required to explain the amount of variance specified
% as PEV (proportion of explained variance). Returns a structure with
% results in its fields.
%
% PARAMETERS
% ----------
% X -- Data taking the form of an m x n matrix where rows correspond to
%      observations and variables to columns. Function will ensure
%      variables are mean centered at zero. B refers to X when
%      mean-centered (columns sum to 0).
% Name-Value Pairs (nv)
%   'corr_mat'        : Must be logical true or false (default). Setting
%                       true will cause PCA to be performed on correlation
%                       rather than covariance matrix.
%   'pev'             : Scalar value that is desired amount of variance
%                       explained by the retained principal components
%                       (scalar value from 0 to 1). Default 0.8.
%   'sign_convention' : Logical true or false corresponding to option to
%                       enforce a sign convention on the right singular
%                       vectors matching that of pca.m from R2019b (largest
%                       component of each vector by absolute magnitude will
%                       be positive). Note that though the column signs are
%                       determined based on the components of the right
%                       singular vectors, the same convention (that is the
%                       same column signs) will be applied to columns of U
%                       as well, so that the reconstruction A = U*S*V'
%                       holds. Default true.
%   'n_dims'          : Option to pass in integer value specifiying number
%                       of principal components to retain. For n_dims = n,
%                       the leading n PCs will be retained (ordered by size
%                       of associated eigenvalue). Will override 'pev'
%                       argument if nonempty. Alternatively, user may
%                       specify 'full', in which case all principal
%                       components with non-zero eigenvalues will be
%                       returned. The user may request more principal
%                       components than there are non-zero eigenvalues
%                       (though calling non-econ svd directly may be more
%                       preferrable in this case). Default empty.
%   'return_all'      : Logical true or false. If true, overrides
%                       everything else and forces all columns of U, S, and
%                       V, (up to the limits imposed by the 'econ' version
%                       of SVD) as well as everything calculated from them,
%                       to be returned. Default false.
% 
% RETURNS
% -------
% out -- 1 x 1 struct with the following fields, where d is generally the
%        number of principal components retained:
%   .U             -- m x d matrix whose columns are the left singular
%                     vectors of Y (mean-centered version of X s.th.
%                     columns sum to 0). d = n at most if all PCs with
%                     nonzero eigenvalues retained and m > n (since
%                     economy-sized SVD is used). If m < n, U is still
%                     generally m x d, though it is now at most m x m (here
%                     the final, ordered left singular vector in this m x m
%                     case will have associated singular value 0 due to
%                     mean-centering; if the mean-centered data matrix is
%                     rank-deficient by more than 1, more than one left
%                     singular vector may have associated singular value 0,
%                     but this is exceedingly rare in empirical
%                     applications due to noise). Equivalent to principal
%                     components normed to be of unit length (recall that
%                     being of unit length is generally not the same thing
%                     as having unit variance).
%   .S             -- n x d (if m > n) diagonal matrix containing the
%                     singular values of B. m x d if m < n. Note that d may
%                     not exceed n if m > n or m if m < n.
%   .V             -- n x d matrix whose columns are the right singular
%                     vectors of B giving the directions of the retained
%                     principal axes in original variable space. Equivalent
%                     to the unit eigenvectors of the
%                     covariance/correlation matrix. d = n if m > n and all
%                     PCs retained. If m < n, d may not exceed m, though
%                     only m - 1 columns of V at most will be associated
%                     with nonzero eigenvalues due to mean-centering (see
%                     'n_dims' and 'return_all' under the Name-Value Pairs
%                     section of PARAMETERS). Again, more than 1 right
%                     singular vector may have associated singular value of
%                     0 if the mean-centered matrix is rank-deficient by
%                     more than 1, though this is rare in empirical cases,
%                     due to the noise in real-world data.
%   .sigmas        -- Vector of singular values corresponding to retained
%                     left/right singular vectors (equivalent to the
%                     diagonal of S).
%   .lambdas       -- Vector of eigenvalues corresponding to retained
%                     principal components (recovered from singular
%                     values).
%   .prin_comps     -- Size(U) matrix that corresponds to coordinates of the
%                     original observations (row vectors) in the basis
%                     given by the retained right singular vectors
%                     (equivalent to unit eigenvectors of the
%                     covariance/correlation matrix). Also sometimes called
%                     empirical orthogonal variables, whose elements are
%                     the "scores" (of each observation on each principal
%                     axis). 
%   .loadings      -- Retained unit eigenvectors/rtSingVecs scaled by
%                     square root of eigenvalues. Components of columns of
%                     Y in the basis given by columns uf U.
%   .eigenspectrum -- All eigenvalues (including those with value 0) of the
%                     covariance/correlation matrix. 
% 
% Author: Jonathan Chien 8/29/20. Version 2.3. Last update 5/3/22.

arguments
    X
    nv.corr_mat = false
    nv.pev (1,1) = 0.8
    nv.sign_convention = true
    nv.n_dims = [];
    nv.return_all = false
end

% Check for and handle presence of NaN type among input data. Remove any
% observations with NaN values (svd function does not tolerate NaNs).
if any(isnan(X), 'all')
    warning(['NaNs present in input data. Removing observations with '...
             'NaN values before calling svd function.'])  
    X(any(isnan(X), 2), :) = [];
end

% Option to normalize mean-centered variables by their respective standard
% deviations and perform PCA based on correlation matrix. If not, at least
% ensure data are mean-centered at origin. Note that either option will
% result in the loss of one degree of freedom. If m < n and matrix has full
% row rank, this will also result in decrease of rank by 1.
if nv.corr_mat, Y = normalize(X); else, Y = X - mean(X); end

% Perform economy-sized singular value decomposition.
[U, S, V] = svd(Y, 'econ');

% Option to enforce sign convention on left and right singular vectors.
if nv.sign_convention
    % Match MATLAB's convention (designed to resolve sign ambiguity) of
    % forcing largest component (by absolute magnitude) of each right
    % singular vector to be positive. Code for this adapted from lines 429
    % to 437 of pca.m in R2019b. Note that one potential issue with this
    % method is that a given singular vector could conceivably have more
    % than one component with opposite signs but the same magnitude w/in
    % numerical error, and one of these components would ostensibly thus be
    % "randomly" selected as largest, though I doubt this would be that big
    % of an issue in practice. Same convention (calculated based on V) is
    % applied to columns of U as well to ensure that the reconstruction Y =
    % U*S*V' holds.
    [~, i_max_comp] = max(abs(V), [], 1);
    [dim1, dim2] = size(V);
    column_sign = sign(V(i_max_comp + (0:dim1:(dim2-1)*dim1)));
    U = bsxfun(@times, U, column_sign); 
    V = bsxfun(@times, V, column_sign);
end

% Extract singular values from the matrix sigma (already sorted in
% descending order).
sigmas = diag(S);

% Degrees of freedom = m-1 if data is centered (as columns now sum to zero)
% and m otherwise.
dof = size(Y, 1) - 1;

% Recover eigenvalues of the covariance/correlation matrix, which are
% equivalent to the variance of the corresponding principal component.
lambdas = sigmas.^2 / dof;
eigenspectrum = lambdas;

% Calculate principal components. Note that USV'V = US = YV.
prin_comps = U*S; 

% "Loadings" in the style of factor loadings, i.e. endowed with variance.
% This is the "synthesis" model of PCA as a special case of unrotated
% factor analysis (namely, where all variance is deemed shared), with
% emphasis placed on weighting a generative set of latent variables, as
% opposed to the "analytic" model.
loadings = V .* sqrt(lambdas)';

% Determine number of principal components needed to achieve specified PEV
% (or use 'return_all' or 'n_dims' options if specified).
if nv.return_all
    i_dim = length(eigenspectrum);
    
elseif strcmp(nv.n_dims, 'full')
    i_dim = min(dof, size(Y, 2));
    
elseif ischar(nv.n_dims) && ~strcmp(nv.n_dims, 'full')
    error("Invalid value for 'n_dims'.")
    
elseif ~isempty(nv.n_dims)
    i_dim = nv.n_dims;
    
else % Use proportion explained variance
    current_pev = 0;
    i_dim = 0;
    total_var = sum(lambdas);
    while current_pev < nv.pev
        if ~ismembertol(current_pev, nv.pev) 
            i_dim = i_dim + 1;
            current_pev = current_pev + lambdas(i_dim) / total_var;
        else
            break
        end
    end
end

% Truncate outputs and store in structure to be returned.
out.U = U(:, 1:i_dim);
out.S = S(:, 1:i_dim);
out.V = V(:, 1:i_dim);
out.sigmas = sigmas(1:i_dim);
out.lambdas = lambdas(1:i_dim);
out.prin_comps = prin_comps(:, 1:i_dim);
out.loadings = loadings(:, 1:i_dim);
out.eigenspectrum = eigenspectrum;

end
