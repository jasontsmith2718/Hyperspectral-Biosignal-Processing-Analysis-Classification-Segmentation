% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% This script is intended to give an example of unmixing hyperspectral
% Raman imaging data and displaying the resulting overlay.
% 
% Jason T. Smith (RPI) & Dr. Lingyan Shi (UCSC), Spring 2020
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% % % % % SECTIONS: % % % % % 
% - Load in data (either from .mat file or TIF sequence)
% - Two-component Principal Component Analysis (PCA) unmixing
% - Hyperspectral phasor analysis.
% - K-means clustering of phasor-retrieved components for segmentation.
% % % % % % % % % % % % % % %

% Make sure path to '.../functions/' folder is added.

%% Save TIFF sequence as MATLAB data file
% **SKIP THIS PART IF YOU DOWNLOADED THE .MAT FILE** %

% Example data in '.../tiffStack_MCF10A-SRS/'
fN = dir('*.tif'); % Go to directory of hyperspectral Raman data (TIF FORMAT)
hsData = zeros([size(imread(fN(1).name)) length(fN)]); % Preallocate
for i = 1:length(fN)
    hsData(:,:,i) = imread(fN(i).name);
end

figure;
imagesc(max(hsData,[],3)); % Check for maximum intensity
title('Maximum Intensity Image')

% Save as file for both MATLAB and python upload.
fileN = '';
save(fileN,'hsData','maxI','-v7.3'); 

%% PCA unmixing

% % %  SRS Image data example:
load MCF10A-803-10-20-spec-24mm-25-51-0.1.mat
% hsData: SRS data (x, y, wavenumber) - (512, 512, 51)
% maxI: maximum intensity image = max(hsData,[],3);
% wv: wavenumbers (cm^-1)

% % Now we can use the pca() function directly on the entire matrix. 

% Example of using PCA to unmix lipids and proteins from hyperspectral
% raman images. What this gives you is two matrices, the feature 'y' 
% and weight 'x' matrix along with the eigenvector column 'z' and the 
% percentage of variance that is accounted for by each principle component 
% 'explained'.

% Reshape data into 2D (for PCA-basesd matrix decomposition)
hsDataR = reshape(hsData,[512*512 51]);

% Normalize data by Euclidean
X_norm = hsDataR' - repmat(mean(hsDataR',2),1,size(hsDataR',2));

% PCA
[x,y,z,a,explained]=pca(X_norm);

% Plot the first ten values of 'explained' in order to visualize the 
% percentage of the total variance explained by these first ten PCs.
figure; bar(explained(1:10), 'r');
xlabel('PC #'); ylabel('Total Variance Contribution (%)');

%% MATLAB figures (seperate & multi-colormap overlay)
% Pick first two PCs and display the images of both (negatives excluded)
x = x';
hsDataRr = reshape(x(1:2,:)',[512 512 2]);
for i = 1:2
    figure; imagesc(max(hsDataRr(:,:,i),0));
    caxis([0 0.01]); colormap(jet);
    title(['PC # ' num2str(i)]);
end
hsDataRr(hsDataRr==0) = NaN; % Assign NaN values for visualization
% Image overlay for PC #1 (lipids) onto PC #2 (proteins)
plotUnmixed(hsDataRr, [0.01 0.01])

%% Phasor analysis augmented with K-means clustering for unsupervised SRS unmixing

% Reshape data to 2D (x*y, wv)
hsDataR = reshape(hsData,[512*512 51]); 

specresol = size(hsDataR,2);
lamda = 1:specresol;

% Spectral phasor transform (first harmonic)
% Good reference: Golfetto O., et al "The Laurdan Spectral Phasor Method to Explore 
% Membrane Micro-heterogeneity and Lipid Domains in Live Cells." (2015) 

% 2D polar representation
g = @(x,y) sum(x.*cos(2*pi*y/specresol))/sum(x); 
s = @(x,y) sum(x.*sin(2*pi*y/specresol))/sum(x);

% Pre-allocate for every Raman spectra's (g, s) pair
gD = zeros(1,size(hsDataR,1));
sD = zeros(1,size(hsDataR,1));

% Iterate through each row (spectra)
for i = 1:size(hsDataR,1)
    gD(i) = g(hsDataR(i,:),lamda); 
    sD(i) = s(hsDataR(i,:),lamda);
end

% Create array of size [# samples, 2] with phasor components for K-means
gs(:,1) = gD;
gs(:,2) = sD;

% % % K-MEANS for unsupervised phasor clustering % % %

% K-means requires the user to choose the number of clusters to search for
% within the data a priori. Let us try just two clusters, like with PCA.

rng(1);
nC = 2;
id = kmeans(gs,nC); % Each pixel assigned a value between [1-nC]

id = reshape(id, [512 512]); % Reshape to original (x, y)

figure; imagesc(id); % Visual
title([num2str(nC) ' Component K-Means Clustering w/ Phasor']);
colormap(jet);

i1 = maxI; % Isolate pixels from first group
i1(id~=2) = 0;

i2 = maxI; % Isolate pixels from second group
i2(id==2) = 0;

% Preallocate for two-cluster (two colormap) overlay
v = zeros([size(id) 2]);
v(:,:,1) = i1;
v(:,:,2) = i2;
v(v==0) = nan; % Assign NaN values for visualization
plotUnmixed(double(v),[3500 3500]); % *Optimize color axis with a second argument*

%% MORE TO BE ADDED... (LAST UPDATE: 03/17/2020)
