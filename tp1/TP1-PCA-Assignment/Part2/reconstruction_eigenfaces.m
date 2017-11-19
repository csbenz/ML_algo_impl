function [reconstructed_faces] = reconstruction_eigenfaces(X, V, Mu, sizeIm)
%RECONSTRUCTION_EIGENFACES Reconstructs face images from lower dimensional
%space to compare the result to the original face image and to the mean face image
%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%       o V      : (M x M), matrix of eigenvectors
%       o Mu     : (N x 1), mean face image
%       o sizeIm : (1 x 1), size of image (width=height)
%
%   output ----------------------------------------------------------------
%
%       o reconstructed_faces : (sizeIm x sizeIm x 4), 4 lossy reconstructions 
%                               of the first image, in image format

% Plot Original Face
subplot(1, 6, 6);
colormap('Gray');
imagesc(reshape(X(:, 1), sizeIm, sizeIm));

% Plot Mean Face
subplot(1, 6, 1);
imagesc(reshape(Mu,sizeIm,sizeIm));

% Plot Projections p={1,51,101,151}
reconstructed_faces = zeros(sizeIm, sizeIm, 4);

i = 2;

for p = 1:50:151
    [A_p,Y] = project_pca(X, Mu, V, p);
    [X_hat] = reconstruct_pca(Y, A_p, Mu);
    
    eigenface = reshape(X_hat(:,1), sizeIm, sizeIm);
    reconstructed_faces(:, :, i - 1) = eigenface;
    
    subplot(1, 6, i);
    imagesc(eigenface);
    title(['p =', num2str(p)]);
    
    i = i + 1;
end


end