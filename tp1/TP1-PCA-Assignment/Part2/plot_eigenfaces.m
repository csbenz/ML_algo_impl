function [V, L, Mu, eigenfaces] = plot_eigenfaces(X, sizeIm)
%PLOT_EIGENFACES Extracts and displays eigenfaces based on dataset X
%   
%   
%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%       o sizeIm : (1 x 1), size of image (width=height)
%
%   output ----------------------------------------------------------------
%
%       o eigenfaces : (sizeIm x sizeIm x 20), First 20 eigenvectors in image format.

% Auxiliary variables
[N, M] = size(X);
N_displayed_images = 20;

% Output variables
V  = zeros(N,N);
L  = zeros(N,N);
Mu = zeros(N,1);

[V, L, Mu] = my_pca(X);

eigenfaces = zeros(sizeIm,sizeIm,N_displayed_images);
%eigenfaces = [];
% Construct EigenFaces
for i = 1:N_displayed_images
    eigenface = reshape(V(:,i), sizeIm, sizeIm);
    
    eigenfaces(:,:,i) = eigenface;
    
    subplot(2, 10, i);
    colormap('Gray');
    imagesc(eigenface);
end

end