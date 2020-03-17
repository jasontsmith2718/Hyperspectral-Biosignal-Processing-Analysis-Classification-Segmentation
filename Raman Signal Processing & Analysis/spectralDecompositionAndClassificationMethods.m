% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%  This script is intended to explain how to use a few MATLAB functions for
%  background-subtraction, normalization and 2-class SVM separation via
%  matrix decomposition.
%  Jason Smith, RPI, 2020
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% % % % % SECTIONS: % % % % % 
% - Prepare data (load, background subtraction, normalization, etc.)
% - Principal Component Analysis (PCA)
% - Non Negative Matrix Factorization (NMF)
% - t-Distributed Stochastic Neighbor Embedding (t-SNE)
% - Multi-Layered Perceptron (MLPs)
% - Autoencoders (MLP AEs)
% % % % % % % % % % % % % % % 

clear all;
close all;
%% Import data & add path

% Make sure path to '.../functions/' folder is added.

% A few of the breast sample spectra obtained for use in the following:

% % % % % % % % % % % % 
% "Characterization and discrimination of human breast cancer and normal 
% breast tissues using resonance Raman spectroscopy", 2018 Feb 19 (Vol. 
% 10489, p. 104890X)
% % % % % % % % % % % % 

load VRRdata_breastExVivo

% VRR.wv = wavenumber (cm^-1)
% VRR.normal = VRR spectra from normal tissue
% VRR.cancer = VRR spectra from cancerous tissue

% Get the Raman-Shift (cm^-1) values (you only want matrices with a spectra
% at each column).
wv = VRR.wv;

spectraN = VRR.normal;
spectraC = VRR.cancer;

[~,b] = size(spectraN);
[~,dec] = size(spectraC);

% Extracted the spectral fingerprint region
Csplit = spectraC(551:2000,:);
Nsplit = spectraN(551:2000,:);
wvSplit = wv(551:2000);

% % Just to check out the raw comparison:
figure;
plot(wvSplit,Csplit(:,1), 'LineWidth', 2.5);
hold on
plot(wvSplit,Nsplit(:,1),'r','LineWidth', 2.5);
legend('Normal','Cancer');
xlabel('Raman Shift (cm^-1)')
ylabel('Intensity (photon count)')

%% Polynomial background correction
% backcor() is a function for background correction, taken from
% file-exchange (open-source). Don't worry about how backcor() works for
% now, just know that it fits a polynomial to the 'fluorescence background'
% and we can subsequently use that fit to get rid of it (native fluorescence noise
% from tissue can be a big problem). Check out the before and after.

% % Using backcor() with a loop through each spectra in each class.
for i = 1:b
BG1 = backcor(linspace(551,2000,1450)',Nsplit(:,i)',12,0.001,'ah');
    N_bg(:,i) = BG1;
    N_rmbg_split(:,i) = Nsplit(:,i) - BG1;
end

for i= 1:dec
BG2 = backcor(linspace(551,2000,1450)',Csplit(:,i)',12,0.001,'ah');
    C_bg(:,i) = BG2;
    C_rmbg_split(:,i) = Csplit(:,i) - BG2;
end


%% A quick comparison of raw vs. backcor()
% Cancerous:
figure;
plot(wvSplit,Csplit(:,1), 'r', 'LineWidth', 2.5)
hold on
plot(wvSplit,C_rmbg_split(:,1),'k','LineWidth', 2.5)
xlabel('Raman Shift (cm^-1)')
ylabel('Intensity (a.u.)')
legend('non-removed', 'removed')
title('Cancerous VRR Ex.')

% Benign:
figure;
plot(wvSplit,Nsplit(:,1), 'b', 'LineWidth', 2.5)
hold on
plot(wvSplit,N_rmbg_split(:,1),'k','LineWidth', 2.5)
xlabel('Raman Shift (cm^-1)')
ylabel('Intensity (a.u.)')
legend('non-removed', 'removed')
title('Benign VRR Ex.')

% As you can see, the 'removed' spectra doesn't have any baseline curvature
% and every intensity peak rises from zero instead of some random point.

% Next step: Concatenation of all spectra (normal and cancerous) into a single matrix. 
% This is a necessary step before using matrix decomposition.

X_rmbg_split = [N_rmbg_split C_rmbg_split]; % (rows, columns) = (wv, samples)

% Now just to normalize by the Euclidean.
X_rmbg_split_norm = X_rmbg_split./repmat(sqrt(sum(X_rmbg_split.^2)),length(Csplit(:,1)),1);

% Plotting all normalized spectra out.
figure;
for i=1:b+dec
    plot(wvSplit,X_rmbg_split_norm(:,i));
    hold on
end
xlabel('Raman Shift (cm^-^1)')
ylabel('Intensity (a.u.)')
title('Background-removed & Normalized (All VRR)');
xlim([750 2000]);

% Plot the average and std of all normalized spectra in each class!

figure;
subplot(2,1,1);
shadedErrorBar(wvSplit, mean(X_rmbg_split_norm(:,1:b),2), std(X_rmbg_split_norm(:,1:b)'),'lineprops','b');
xlim([750 2000]); xlabel('Raman Shift (cm^-^1)'); ylabel('Intensity (a.u.)');
title('Benign VRR (Average +/- STD, Processed)')

subplot(2,1,2);
shadedErrorBar(wvSplit, mean(X_rmbg_split_norm(:,b+1:end),2), std(X_rmbg_split_norm(:,b+1:end)'),'lineprops','r');
xlim([750 2000]); xlabel('Raman Shift (cm^-^1)'); ylabel('Intensity (a.u.)');
title('Cancerous VRR (Average +/- STD, Processed)')

% % % % % % **FileExchange function** % % % % % % 
% shadedErrorBar(x,y,std,...) 
% https://www.mathworks.com/matlabcentral/fileexchange/26311-raacampbell-shadederrorbar
% % % % % %

%% Spectral decomposition
% There a good amount of matrix decomposition techniques used in the field
% today, but there are two that stick out as being the most conventionally used.
% These are:

% % 1) Principal Component Analysis (PCA)

% % 2) Non-Negative Matrix Factorization (NMF)

% The main differences between these two are:
% I) PCA doesn't introduce any non-linearity into the solution space.
%
% II) NMF is a machine-learning technique (starts at a randomized position
% in the feature-space, attempts to find an optimal local minima). In contrast,
% PCA gives an exact/unique solution every time it is run.
%
% III) NMF is positively constrained (hence the name). This introduces a much
% greater degree of sparsity into the solution obtained in contrast to PCA, which 
% is preferable depending on the application.

%% Perform the matrix decomp via PCA()
% Now that we have the spectra in a processed and normalized form, we can
% begin our approach. First, we can try PCA. PCA requires the data to be
% normalized around zero, which you do using the following line.
X_norm = X_rmbg_split_norm - repmat(mean(X_rmbg_split_norm,2),1,size(X_rmbg_split_norm,2));

% Now we can use the pca() function directly on the entire matrix. What this
% gives you is two matrices, the feature 'y' and weight 'x' matrix along
% with the eigenvector column 'z' and the percentage of variance that is
% accounted for by each principle component 'explained'.

[x,y,z,a,explained]=pca(X_norm);

% Plot the first ten values of 'explained' in order to visualize the 
% percentage of the total variance explained by these first ten PCs.
figure; bar(explained(1:10));
xlabel('PC #'); ylabel('Total Variance Contribution (%)');

% Out of personal preference, I plot the weight values in this way (by
% transposing x first and taking the first three rows).
x = x';
x1 = x([1 2 3],1:b)'; % PC loadings 1-3 for normal group
x2 = x([1 2 3],b+1:end)'; % PC loadings 1-3 for cancerous group

% So, we can see that x is now of size (Spectra# x Spectra#), or, reduced 
% significantly from its original by taking the first three component values
figure;
scatter3(x1(:,1),x1(:,2),x1(:,3),'kd','SizeData',50, 'MarkerFaceColor', 'b');
hold on
scatter3(x2(:,1),x2(:,2),x2(:,3),'ko','SizeData',50, 'MarkerFaceColor', 'r');
legend('Benign','Cancerous');
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');

figure;
plot(wvSplit,y(:,1), '-b', 'LineWidth', 1.5)
hold on
plot(wvSplit,y(:,2), '-k', 'LineWidth', 1.5)
hold on
plot(wvSplit,y(:,3), '-m', 'LineWidth', 1.5)
legend('PC1','PC2','PC3');
xlabel('Raman-shifted wavenumber (cm^-^1)');
ylabel('Intensity');
title('Top Three Principle Components Overlay');
xlim([750 2000]);

% % % 
% Now to perform leave-one-out cross validation via three-feature Support
% Vector Machine (SVM) to obtain the sensitivity, specificity and accuracy.

loocvclassify(x1, x2);

% % % % % % **Reference** % % % % % % 
% loocvclassify(group#1, group#2) 
% Author: Dr. Binlin Wu, Southern Connecticut State University
% % % % % %

%% Perform the matrix decomp via nnmf()
% In contrast to PCA, NMF input should not be normalized about the mean.
% Choosing r = 10 (10 "feature" spectra)
[W,H]=nnmf(X_rmbg_split_norm,10);

% W is non-square so we must use the Moore–Penrose inverse 'pinv()' to
% obtain the NMF "loadings" from each spectral sample.
X_NMF=pinv(W)*X_rmbg_split_norm;

% As you'll see, matrix A has 10 feature spectra. You can choose 1 or 2 of these
% and see how it effects your classification performance.

f1 = 2;
f2 = 3;

x1 = X_NMF([f1 f2],1:b)';
x2 = X_NMF([f1 f2],b + 1:end)';

figure;
scatter(x1(:,1),x1(:,2),'kd','SizeData',50, 'MarkerFaceColor', 'b');
hold on
scatter(x2(:,1),x2(:,2),'ko','SizeData',50, 'MarkerFaceColor', 'r');
legend('Benign','Cancerous');
xlabel(['NMF #' num2str(f1)]);
ylabel(['NMF #' num2str(f2)]);

% % % % 2-Feature SVM (can use > 2 as well)
% svm2d(x1, x2);

% % % % % % **Reference** % % % % % % 
% svm2d(group#1, group#2) 
% Author: Dr. Binlin Wu, Southern Connecticut State University
% % % % % %

% LOOCV for quantification
loocvclassify(x1,x2);

% NOTE THAT NMF IS A ITERATIVE, ERROR MINIMIZATION TECHNIQUE! Thus, NMF will 
% *NOT* always give the same solution and one should run it a few times to make
% sure they get something more or less stable.

% % % % Plot the NMF-extracted component spectra
figure;
plot(wvSplit, W(:,f1), '-b', 'LineWidth', 1.5);
hold on;
plot(wvSplit, W(:,f2), '-k', 'LineWidth', 1.5);
legend(['NMF Feature #' num2str(f1)], ['NMF Feature #' num2str(f2)])
xlim([750 2000]);


%% t-SNE
% I'm including this because t-SNE is a very clever unsupervised technique
% for projecting data possessing enormous feature sizes to 2D or 3D. Here
% are two sites where you can get a nice overview on this.
% I) https://www.youtube.com/watch?v=NEaUSP4YerM
% II) https://lvdmaaten.github.io/tsne/

% So, in this case you need to have your matrix in size (samples x
% features), so we transpose. This algorithm is unique for three main
% reasons:

% 1) One will never get the same solution twice (without a set random seed).
% 2) Distance in t-SNE space means nothing. A cluster that seems close one
% run could be crazy far apart another run and there isn't any way to use
% the resulting coordinates for a SVM to quantify it.
% 3) t-SNE works not only with groups, but with regression.

% So, here is an example of running it. Check the MATLAB documentation if
% you're interested. Yes, it takes a while (especially with 'NumDimensions'
% set to 3).

% 'Y' will be the (#Samples x 3) t-SNE solution
Y = tsne(X_rmbg_split_norm','Algorithm','exact','Distance','cosine','NumDimensions',3);

% 3D t-SNE plot with normal and cancerous labeled accordingly.
figure; scatter3(Y(1:b,1),Y(1:b,2),Y(1:b,3),'kd','SizeData',50, 'MarkerFaceColor', 'b')
hold on; scatter3(Y(b+1:end,1),Y(b+1:end,2),Y(b+1:end,3),'ko','SizeData',50, 'MarkerFaceColor', 'r')
legend('Benign','Cancerous');
xlabel('t-SNE_1');
ylabel('t-SNE_2');
zlabel('t-SNE_3');

% The only "problem" with t-SNE is that it is qualitative. One cannot use
% SVM like we did before on the t-SNE extracted features given that
% distance in the t-SNE space means nothing. Nevertheless, it has been use for 
% strong qualitative support across biosignal and bioimage processing studies.


%% Create Multi-Layer Perceptron for classification!
% MLPs are the simplest form of "deep learning". MATLAB's toolbox is okay
% for some applications, but one needs to have the Neural Network Toolbox
% downloaded.

rng(1); % Random seed
targetN = [ones(b,1) zeros(b,1)]'; % One-hot encoding for benign class
targetC = [zeros(dec,1) ones(dec,1)]'; % One-hot encoding for cancerous class
targets = [targetN targetC]; % Combine for labels of all samples!

% Create two-hidden layer MLP for VRR-based tissue discrimination
clear net;
net = patternnet([75,15]); % Two hidden layers (75 and 15 nodes each)
net.trainParam.epochs = 150;
net.trainParam.max_fail = 25;
net.trainParam.min_grad = 0;
net.layers{1}.transferFcn = 'tansig'; % tanh() activation
net.layers{2}.transferFcn = 'tansig'; % tanh() activation
net.layers{3}.transferFcn = 'logsig'; % sigmoid output
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 60/100; % 60% training
net.divideParam.valRatio = 10/100; % 10% validation
net.divideParam.testRatio = 30/100; % 30% test
net.trainFcn = 'trainscg'; % Optimization technique
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

% Train MLP
[net,tr] = train(net,X_rmbg_split_norm,targets);

% See how the MLP performs on test data (data never "seen" by network)
testD = X_rmbg_split_norm(:,tr.testInd); % Test VRR input
testGT = targets(:,tr.testInd); % Test G.T.
y = net(testD); % Use MLP on test VRR

% Built-in MATLAB functions to plot the loss curves, classification results 
% on the test data (both confusion matrix & ROC curves);

figure; 
plotperform(tr); % loss

figure; 
plotconfusion(testGT,y); % confusion

figure; 
plotroc(testGT,y); % ROC


%% Autoencoder MLPs
% Autoencoders work via two major parts:
% 1) An encoder, which squashes the input down to a significantly reduced 
% dimensionality, and...
% 2) A decoder, which attempts to map a squashed representation back to the
% original input spectra.
%
% - Thus, AEs are often used as an unsupervised technique for data reduction.
%
% - They can be used for spectral decomposition in similar fashion as the
% techniques above, and the features obtained from feeding the input data
% into the encoder can be used for clustering.
%
% % NOTE: **If you really want to use this technique, I would strongly
% recommend the use of Python and Tensorflow over MATLAB's toolbox.** % %

rng(1); % Random seed
AEnodes = 10; % # Nodes in the hidden layer of the AE

% Initialize AE 
autoenc = trainAutoencoder(X_rmbg_split_norm,AEnodes,'MaxEpochs',200,...
'L2WeightRegularization',0.001,... % for L2 loss term
'EncoderTransferFunction','satlin',... % Hidden layer #1 activation
'DecoderTransferFunction','logsig'); % Output activation

XReconstructed = predict(autoenc,X_rmbg_split_norm);
mseError = mse(X_rmbg_split_norm-XReconstructed);
disp(['MSE AE: ' num2str(mseError)]);

% Feed VRR data into the AE's encoder to get 10 squashed features for each
% sample!

Z = encode(autoenc,X_rmbg_split_norm); % Encoding w/ trained AE

% Which features? ( Any numbers between 1 - AEnodes )
f1 = 1;
f2 = 2;
f3 = 3;

x1 = Z([f1 f2 f3],1:b)'; % Pick out normal class
x2 = Z([f1 f2 f3],b + 1:end)'; % Pick out cancerous class

figure;
scatter3(x1(:,1),x1(:,2),x1(:,3),'kd','SizeData',50, 'MarkerFaceColor', 'b');
hold on
scatter3(x2(:,1),x2(:,2),x2(:,3),'ko','SizeData',50, 'MarkerFaceColor', 'r');
legend('Benign','Cancerous');
xlabel(['Autoencoder F#' num2str(f1)]);
ylabel(['Autoencoder F#' num2str(f2)]);
zlabel(['Autoencoder F#' num2str(f3)]);

% Trained AE has matrices with weights corresponding to VRR spectra in both
% the encoder and decoder. Can we interpret these?

% Trained encoder: size (AEnodes, wv) 
enc = autoenc.EncoderWeights;
figure; plot(wvSplit, enc(f1,:), 'LineWidth', 1.5);
hold on; plot(wvSplit, enc(f2,:), 'LineWidth', 1.5);
hold on; plot(wvSplit, enc(f3,:), 'LineWidth', 1.5);
xlabel('Raman Shift (cm^-^1)'); ylabel('Intensity (a.u.)');
legend(['AE Encoder #' num2str(f1)], ['AE Encoder #' num2str(f2)], ['AE Encoder #' num2str(f3)])
xlim([750 2000]);

% Trained decoder: size (wv, AEnodes) 
dec = autoenc.DecoderWeights;
figure; plot(wvSplit, dec(:,f1), 'LineWidth', 1.5);
hold on; plot(wvSplit, dec(:,f2), 'LineWidth', 1.5);
hold on; plot(wvSplit, dec(:,f3), 'LineWidth', 1.5);
xlabel('Raman Shift (cm^-^1)'); ylabel('Intensity (a.u.)');
legend(['AE Decoder #' num2str(f1)], ['AE Decoder #' num2str(f2)], ['AE Decoder #' num2str(f3)])
xlim([750 2000]);

%% MORE TO BE ADDED... (LAST UPDATE: 03/17/2020)