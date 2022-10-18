# Data_generation description
##### MSc Thesis - Sonia Laguna - ETH Zurich
This file includes the description of the data generation approaches and files used, found in the folder `codes\data_generation`. 
The directories found are the following: 
- `uncertainty_project_sonia`: New Matlab codes for data generation created in this project
    -  `clinical`: Codes used to generate clinical data reconstructions. `mask_generator` includes files to generate the masks for point annotations. `mat_bin_convert`includes the codes to create mat files from the original bin raw recordings. `stacking_clinical.m` is used to generate the final files with all correpsonding frames.   
    - `MS_data` : Codes used to work with Multistatic data. The files include the code to create the LBFGS data and new ray-based MS test set and the ray-based varying contrast data. `displacement_generation` was created to recompute the displacement estimates and correlation coefficients for in-vivo phantom and clinical data and fullpipeline. 
    - `VS_data`: Codes to generate the LBFGS of clinical data and of simulated data, together with the displacement estimates of the new VS ray-based test set. 
- `data_generation_Melanie`: Codes used in the previous Variational Network project to generate the Ray-Based Mulstistatic synthetic data. 
-`SimulationModels-melTest`: Codes used in the previous Variational Network project to generate the Fullpipeline Mulstistatic synthetic data. 
- `Genertic_Data_Processing_Structure-master`: Codes developed in previous work to create displacement estimates coming from different sequences, Multistatic and Plane Waves. 
- `jason_reconstruction`: Codes used by previous student to create VS synthetic data. 
- `Precomp_Distance_Matrix`: Distance Matrices used in the displacement estimation and inverse problem
