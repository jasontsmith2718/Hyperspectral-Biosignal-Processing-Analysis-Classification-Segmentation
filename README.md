# Hyperspectral Biosignal Processing, Analysis, Classification & Segmentation
Hyperspectral signal & image analysis for biological application

--------------------------------------------------------------

This GitHub contains relevant script, data and instructions for:
1. [**'/Raman Signal Processing & Analysis'**](https://github.com/jasontsmith2718/Hyperspectral-Biosignal-Processing-Analysis-Classification-Segmentation/tree/master/Raman%20Signal%20Processing%20%26%20Analysis) (**MATLAB**)

  * Prepare data (load, background subtraction, normalization, etc.)
  
  * Principal Component Analysis (PCA)
  
  * Non Negative Matrix Factorization (NMF)
  
  * t-Distributed Stochastic Neighbor Embedding (t-SNE)
  
  * Multi-Layered Perceptron (MLPs)
  
  * Autoencoders (MLP-AEs)

2. [**Stimulated Raman Spectroscopy (SRS) Imaging Analysis**](https://github.com/jasontsmith2718/Hyperspectral-Biosignal-Processing-Analysis-Classification-Segmentation/tree/master/SRS%20Image%20Processing%20%26%20Analysis). (**MATLAB**)

  * Load in data (either from [.mat](https://drive.google.com/open?id=1iK1yR8uKoaBwu6BHpe3jn5ZyZM3ea280) file or [TIF](https://github.com/jasontsmith2718/Hyperspectral-Biosignal-Processing-Analysis-Classification-Segmentation/blob/master/SRS%20Image%20Processing%20%26%20Analysis/tiffStack_MCF10A-SRS.zip) sequence)
  
  * Two-component Principal Component Analysis (PCA)-based unmixing & segmentation.
  
  * Hyperspectral phasor analysis.
  
  * K-means clustering of phasor-retrieved components for segmentation.
  
  ![SRS Cell Segmentation via PCA](https://i.imgur.com/fgU7cON.png)
  
#### _**THIS WORK IS ONGOING**_

--------------------------------------------------------------

### Relevant Data files:

1. **Visible Resonance Raman (VRR)** [spectra](https://github.com/jasontsmith2718/Hyperspectral-Biosignal-Processing-Analysis-Classification-Segmentation/blob/master/Raman%20Signal%20Processing%20%26%20Analysis/VRRdata_breastExVivo.mat) acquired of human breast tissue samples _ex vivo_. [1]

2. **Stimulated Raman Spectroscopy (SRS)** [image](https://drive.google.com/open?id=1iK1yR8uKoaBwu6BHpe3jn5ZyZM3ea280) _in vitro_.

--------------------------------------------------------------

### References & Related Work:

[1] Wu B, Smith J, Zhang L, Gao X, Alfano RR. ["_Characterization and discrimination of human breast cancer and normal breast tissues using resonance Raman spectroscopy_"](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10489/104890X/Characterization-and-discrimination-of-human-breast-cancer-and-normal-breast/10.1117/12.2288094.full?sessionGUID=d883c9d9-02bc-9993-ced2-68bead49a285&webSyncID=0ce46e9e-6ec7-a49d-ab6a-0cbad059329a&sessionGUID=d883c9d9-02bc-9993-ced2-68bead49a285&SSO=1). Optical Biopsy XVI: Toward Real-Time Spectroscopic Imaging and Diagnosis 2018 Feb 19 (Vol. 10489, p. 104890X). International Society for Optics and Photonics.

[2] Xue J, Pu Y, Smith JT, Gao X, Wang C, Wu B. [" Identifying metastatic ability of prostate cancer cell lines using native fluorescence spectroscopy and machine learning methods "](https://www.nature.com/articles/s41598-021-81945-7). Scientific Reports. 2021 Jan 26. PMID: 33500529

[2] Bendau E, Smith J, Zhang L, Ackerstaff E, Kruchevsky N, Wu B, Koutcher JA, Alfano R, Shi L. ["_Distinguishing Metastatic Triple Negative Breast Cancer from Non‐metastatic Breast Cancer using SHG Imaging and Resonance Raman Spectroscopy_"](https://onlinelibrary.wiley.com/doi/abs/10.1002/jbio.202000005). Journal of Biophotonics. 2020 Mar 26. PMID: 32219996
