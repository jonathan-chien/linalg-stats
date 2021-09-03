function out = pca_jmc(X,nvp)
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
% Name-Value Pairs (nvp)
%   'corrMat'        -- Must be logical true or false (default). Setting
%                       true will cause PCA to be performed on correlation
%                       rather than covariance matrix.
%   'PEV'            -- Scalar value that is desired amount of variance
%                       explained by the retained principal components
%                       (scalar value from 0 to 1). Default 0.8.
%   'signConvention' -- Logical true or false corresponding to option to
%                       enforce a sign convention on the right singular
%                       vectors matching that of pca.m from R2019b (largest
%                       component of each vector by absolute magnitude will
%                       be positive). Note that though the column signs are
%                       determined based on the components of the right
%                       singular vectors, the same convention (that is the
%                       same column signs) will be applied to columns of U
%                       as well, so that the reconstruction A = U*S*V'
%                       holds. Default true.
%   'nDims'          -- Option to pass in integer value specifiying number
%                       of principal components to retain. For nDims = n,
%                       the leading n PCs will be retained (ordered by size
%                       of associated eigenvalue). Will override PEV
%                       argument if nonempty. Alternatively, user may
%                       specify 'full', in which case all principal
%                       components with non-zero eigenvalues will be
%                       returned. Default empty.
%   'returnAll'      -- Logical true or false. If true, overrides
%                       everything else and forces all columns of U, S, and
%                       V, (up to the limits imposed by the 'econ' version
%                       of SVD) as well as everything calculated from them,
%                       to be returned. Default false.
%   'checkRank'      -- Logical true or false. If true, function will check
%                       rank of the mean-centered data matrix and issue a
%                       warning if the user has requested, through the
%                       'nDims' name-value pair, more principal components
%                       than there are non-zero eigenvalues (it is often
%                       useful to work with singular vectors whose
%                       associated singular values = 0, though calling
%                       non-econ svd directly may be more preferrable in
%                       those cases). Since checking the rank involves an
%                       additional call to the svd function, it may be
%                       desirable to suppress this behavior by setting
%                       'checkRank' to false, e.g., if pca_jmc is called
%                       many times in a large loop.
% 
% RETURNS
% -------
% out -- 1 x 1 struct with the following fields, where d is generally the
%        number of principal components retained:
%   .U             -- m x d matrix whose columns are the left singular
%                     vectors of B (mean-centered version of X s.th.
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
%                     'nDims' and 'returnAll' under the Name-Value Pairs
%                     section of PARAMETERS). Again, more than 1 right
%                     singular vector may have associated singular value of
%                     0 if the mean-centered matrix is rank-deficient by
%                     more than 1, though this is rare in empirical cases,
%                     due to the noise in real-world data.
%   .sigmas        -- Vector of singular values corresponding to retained
%                     left/right singular vectors (equivalent to the
%                     diagonal of S).
%   .eigenvalues   -- Vector of eigenvalues corresponding to retained
%                     principal components (recovered from singular
%                     values).
%   .prinComps     -- Size(U) matrix that corresponds to coordinates of the
%                     original observations (row vectors) in the basis
%                     given by the retained right singular vectors
%                     (equivalent to unit eigenvectors of the
%                     covariance/correlation matrix). Also sometimes called
%                     empirical orthogonal variables, whose elements are
%                     the "scores" (of each observation on each PC). May be
%                     calculated as BV (where B is the mean-centered input
%                     data) or as US (U scaled by S).
%   .loadings      -- Retained unit eigenvectors/rtSingVecs scaled by
%                     square root of eigenvalues, analogous to SV (V scaled
%                     by S), but note norming by DOF when calculating
%                     eigenvalues from singular values.
%   .eigenspectrum -- All eigenvalues (including those with value 0) of the
%                     covariance/correlation matrix. 
% 
% Author: Jonathan Chien 8/29/20. Version 2.3. Last update 8/16/21.

arguments
    X
    nvp.corrMat = false
    nvp.PEV (1,1) = 0.8
    nvp.signConvention = true
    nvp.nDims = [];
    nvp.returnAll = false
    nvp.checkRank = true
end

% Check for and handle presence of NaN type among input data.
if sum(isnan(X), 'all') > 0
    warning(['NaNs present in input data. Removing observations with '...
             'NaN values before calling svd function.'])
    % Remove any observations with NaN values (svd function does not
    % tolerate NaNs).
    X(sum(isnan(X), 2) > 0, :) = [];
end

% Ensure data is centered at origin if not already centered. Note this will
% result in the loss of one degree of freedom. If m < n and matrix, this
% will usually also result in decrease of rank by 1.
B = X - mean(X);

% Option to normalize mean-centered variables by their respective 2-norms
% and perform PCA based on correlation matrix.
if nvp.corrMat, B = B ./ vecnorm(B); end

% Perform economy-sized singular value decomposition.
[U, S, V] = svd(B, 'econ');

% Option to enforce sign convetion on left and right singular vectors.
if nvp.signConvention
    % Match MATLAB's convention (designed to resolve sign ambiguity) of
    % forcing largest component (by absolute magnitude) of each right
    % singular vector to be positive. Code for this adapted from lines 429
    % to 437 of pca.m in R2019b. Note that one potential issue with this
    % method is that a given singular vector could conceivably have more
    % than one component with opposite signs but the same magnitude w/in
    % numerical error, and one of these components would ostensibly thus be
    % randomly selected as largest, though I doubt this would be that big
    % of an issue in practice. Same convention (calculated based on V) is
    % applied to columns of U as well to ensure that the reconstruction B =
    % U*S*V holds.
    [~,maxCompIdx] = max(abs(V), [], 1);
    [dim1, dim2] = size(V);
    columnSign = sign(V(maxCompIdx + (0:dim1:(dim2-1)*dim1)));
    U = bsxfun(@times, U, columnSign); 
    V = bsxfun(@times, V, columnSign);
end

% Extract singular values from the matrix sigma (already sorted in
% descending order).
sigmas = diag(S);

% Degrees of freedom = m-1 if data is centered (as columns now sum to zero)
% and m otherwise.
dof = size(B, 1) - 1;

% Recover eigenvalues of the covariance/correlation matrix, which are
% equivalent to the variance explained by the corresponding principal
% component. Note that if using covariance, we must norm the square of the
% singular values by (n-1), as the covariance matrix = B'*B/(n-1), but the
% (n-1) cancels out when dividing by stddev in the case of correlation
% (this is also why cosine similarity, which does not involve the term n, =
% Pearson's correlation coefficient).
if ~nvp.corrMat
    eigenvalues = sigmas.^2 / dof; 
else
    eigenvalues = sigmas.^2;
end
eigenspectrum = eigenvalues;

% Calculate principal components. The "direct" method is US. However, BV is
% also possible (change of basis matrix is on the right and appears
% "untransposed" because we regard row vectors of B as observations; so for
% D = B', and V = eigenvectors of DD', Z = V'D has principal components as
% rows, and Z' = D'V'' = BV, where V is still the matrix of eigenvectors of
% B'B = DD'). There is no need to get too caught up about the syntax of
% rows as observations vs variables, however. SVD gives us information
% about both the rows and columns of X, and barring mean-centering,
% transposition of X merely causes U and V to exchange
% places/interpretations.
prinComps = U*S; 

% "Loadings" in the style of factor loadings, i.e. endowed with variance.
% This is the "synthesis" model of PCA as a special case of factor analysis
% (namely, where all variance is deemed shared), with emphasis placed on
% weighting a generative set of latent variables, as opposed to the
% "analytic" model.
loadings = V .* sqrt(eigenvalues)';

% Determine number of principal components needed to achieve specified PEV
% (or use 'returnAll' or 'nDims' options if specified).
if nvp.returnAll
    % The rank.m function uses SVD. If calling pca_jmc in a loop with
    % large number of iterations, suppressing this may increase speed.
    if nvp.checkRank && length(eigenspectrum) > rank(B) 
        warning(['Requested number of principal components exceeds '...
                 'the rank of the mean-centered matrix. This means '...
                 'that principal components with associated eigenvalue = 0 ' ...
                 'will be returned.'])    
    end
    iDim = length(eigenspectrum);
    
elseif ischar(nvp.nDims) && strcmp(nvp.nDims, 'full')
    iDim = min(dof, size(B, 2));
    
elseif ischar(nvp.nDims) && ~strcmp(nvp.nDims, 'full')
    error("Invalid value for 'nDims'.")
    
elseif ~isempty(nvp.nDims)
    if nvp.checkRank && nvp.nDims > rank(B) 
        warning(['Requested number of principal components exceeds '...
                 'the rank of the mean-centered matrix. This means '...
                 'that principal components with associated eigenvalue = 0 ' ...
                 'will be returned.'])    
    end
    iDim = nvp.nDims;
    
else % use PEV
    currentPEV = 0;
    iDim = 0;
    totalVar = sum(eigenvalues);
    while currentPEV < nvp.PEV
        % In case of floating point error (if currentPEV cannot be
        % represented in binary form):
        if ~ismembertol(currentPEV, nvp.PEV) 
            iDim = iDim + 1;
            currentPEV = currentPEV + eigenvalues(iDim) / totalVar;
        else
            break
        end
    end
end

% Truncate outputs and store in structure to be returned.
out.U = U(:, 1:iDim);
out.S = S(:, 1:iDim);
out.V = V(:, 1:iDim);
out.sigmas = sigmas(1:iDim);
out.eigenvalues = eigenvalues(1:iDim);
out.prinComps = prinComps(:, 1:iDim);
out.loadings = loadings(:, 1:iDim);
out.eigenspectrum = eigenspectrum;

end