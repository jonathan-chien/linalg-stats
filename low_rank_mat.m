function rrMat = low_rank_mat(lowRank,nRows,nCols,nvp)
% Creates an m x n matrix of rank r, where m, n, and r are supplied
% arguments. This is accomplished by creating a random m x n standard
% gaussian matrix, performing SVD, and setting z singular values to 0
% (where z = the difference between the rank of the gaussian matrix and the
% desired value to which we wish to reduce the rank) before finally
% reconstructing the matrix from U, S, and V' (where S has had z singular
% values set to 0). For a matrix of full rank k, we have the low rank
% matrix as [s1*u1*v1' + s2*u2*v2' + ... + sk*uk*vk'], where u and v are
% corresponding left and right singular vectors respectively, and where of
% the k singular values [s1, s2, ... sk], z have been set to 0 (note that
% since each u is a column vector and each v' is a row vector, we obtain
% from each pair an outer product which is unit rank matrix, and the
% reconstructed matrix is assembled as the sum of these unit rank matrices,
% each of which is scaled by its respective singular value, z of which have
% been set to the zero, causing those z unit-rank matrices scaled by zero
% to become the zero matrix). This is essentially dimensionality reduction,
% and barring the lack of mean-centering, corresponds closely to a PCA
% model where the rank of the data and their covariance matrix can be
% reduced by z by setting z eigenvalues to 0 (whereupon the z respective
% principal components have 0 variance--often it is the trailing
% eigenvalues that are set to 0 in PCA applications). Note that this
% function by default overlays a random standard gaussian matrix on top of
% the low rank matrix to mimic noise, creating a matrix that is technically
% full rank but "close" to being low rank, with this "closeness" depending
% essentially on the strength of the noise (controlled through the
% 'noiseFactor' name-value pair). Only if the 'noiseFactor' is set to 0
% will the matrix be truly low rank.
%
% PARAMETERS
% ----------
% lowRank -- Scalar value that is desired (reduced) rank of the matrix.
%            Note that if 'noiseFactor' ~= 0, lowRank will not be the true
%            rank of rrMat.
% nRows   -- Size of first output array dimension.
% nCols   -- Size of second output array dimension.
% Name-Value Pairs (nvp)
%   'remove'      -- String value 'randperm', 'leading', or 'trailing'.
%                    Specifies which z singular values to set to 0. If
%                    'randperm', z singular values are selected randomly
%                    and set to 0. If 'leading', the largest z singular
%                    values are set to 0, and if 'trailing', the smallest z
%                    singular values are set to 0.
%   'noiseFactor' -- Scalar value that multiples the variance of the
%                    standard gaussian distribution overlaid over the
%                    reduced rank matrix to mimic noise. If set to 0, the
%                    matrix will be truly low rank, otherwise it will only
%                    be "close" to being low rank.
%   'affine'      -- Option to translate the point cloud away from the
%                    origin, which may be useful for mimicing neural firing
%                    rate data. If desired, supply a 2-vector whose first
%                    element is a scalar factor applied to the whole matrix
%                    and whose second element is a translation vector
%                    applied to each dimension ([3 11] seems to give
%                    reasonable results). Note that this will inflate the
%                    rank of the matrix by one, but the intrinsic
%                    dimensionality of the manifold should remain
%                    unchanged. If this behavior is not desired, set
%                    'affine' to logical false or an empty vector.
%
% RETURNS
% -------
% rrMat -- m x n matrix of rank r, optionally overlaid with gaussian noise,
%          in which case the matrix will be technically full rank but
%          "close" to being of rank r.
%
% Author: Jonathan Chien Version 1.0. 7/11/21. Last edit: 7/12/21.
% Version history:
%   -- Adapted from a local function in conditions_vs_neurons script in the
%      Tests_demos directory.

arguments
    lowRank (1,1) 
    nRows (1,1)
    nCols (1,1)
    nvp.remove = 'randperm'
    nvp.noiseFactor = 1
    nvp.affine = false % logical false, or suggested: [3 11]
end

% Create full rank standard normal matrix (frMat) and perform SVD. For
% nonsquare matrices, full rank means the highest possible (despite the
% linear dependence of the rows or columns in the case of underdetermined
% and overdetermined systems, respectively).
frMat = randn(nRows, nCols); 
fullRank = min(nRows, nCols);
[U, S, V] = svd(frMat);

% Set singular values to 0.
switch nvp.remove
    case 'randperm'
        remove = randperm(fullRank, fullRank-lowRank);
    case 'leading'
        remove = 1 : fullRank - lowRank;
    case 'trailing'
        remove = lowRank+1 : fullRank;
end
S(sub2ind([nRows nCols], remove, remove)) = 0;

% Reconstruct matrix with reduced rank (rrMat).
rrMat = U * S * V';

% Option to translate point cloud away from origin (more closely mimicking
% real neural firing rates). Note that in so doing we add a degree of
% freedom (column and row means are no longer centered at 0) and inflate by
% 1 the rank of the noiseless data when considered as residing within a
% proper vector subspace. The intrinsic dimensionality of the affine
% space/linear manifold remains unchanged however and will be revealed if
% we "return" this degree of freedom by mean-centering in the
% pre-processing phase of PCA, for in so doing we bring the origin back
% into the manifold, dispensing with the extra dimension that arose to
% accomodate it in the erstwhile vector space.
if nvp.affine
    rrMat = rrMat * nvp.affine(1) + nvp.affine(2); 
end

% Overlay Gaussian noise on top of reconstructed matrix.
rrMat = rrMat + randn(size(rrMat)) * nvp.noiseFactor;
    
end