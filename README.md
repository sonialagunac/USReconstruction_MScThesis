# Uncertainty Estimation in Deep Image Reconstruction using Variational Networks
##### MSc Thesis - Sonia Laguna - ETH Zurich
This file includes the description to use the Variational Network to reconstruct SoS, as well as to use three methods of uncertainty estimation described in the MSc Thesis report. 
The steps to analyse the results and compute the final statistics are included in this document too. 
### 0. Environment setup prior to any experiment.
Set your PYTHONPATH variable to where your code folder is: (i.e. `export PYTHONPATH="${PYTHONPATH}:/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/"`)

Set your DATA_PATH to where your data lies (`export DATA_PATH="/scratch_net/biwidl307/sonia/data_original/"`)

Set your EXP_PATH to where you want to save the experiments (`export EXP_PATH="/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/runs/"`)

The file  "environment.yml" includes the dependencies and channels needed to run this Variational Network. 

### 1. Datasets
This network is trained on Fullpipeline data and Ray-based data. All the data for both Multistatic and Virtual Source sequences can be found in the repository `data_original` with the explanation of all of it in `README_data.md`. 
Additional explanations and codes on how to generate it are found in the folder `/codes/data_generation` in the `README_generation.md`. 

### 2. Train reconstruction network (VN).
 To train the network, the file `train_vn_sos.py` is used and the overall architecture can be found in `vn_net_sos.py`. The config file is a `.yaml` file containing the parameters of the network and of the training. The config file has to be placed in the `configs` folder. Config files of the experiments presented in the thesis are available in this folder. The files ending in "VS" are tailored to train the VN with Virtual Source data and the rest are used for Multistatic data. 
The files including "drop" perform MC dropout uncertainty estimation, those with "bay" perform Bayesian Variational Inference uncertainty estimation and those with "ens" are plain VN with different architectures used in ensembles.
The description of the parameters in the config files included in this project and those previously existing are described in Section 3 below. 
Run `python codes/VNSolver/train_vn_sos.py -config_file '{config_file.yaml}'` to train the VN.

### 3. Evaluate the trained VN on the different test sets. 
To evaluate the network under different noise, undersampling rate, data sources use `evaluation.py` along with the config file used for training, the checkpoint number to restore and a csv file containing the evaluation parameters. 
The csv file is defined in the config file and it is placed in the `evaluation/csv` folder. This csv should include which evaluation experiment you wish to perform, several examples are readily included in the folder. The number of samples of each frame at inference for each uncertainty estimation method can be defined from the config file.
Run `python code/evaluation/evaluation.py -config_file {'config_file_from_training.yaml'} -cpkt {'vn-number'}` for inference.
The description of the config file parameters can be seen below. 

###### Adapted from Melanie Bernhardt
| Name | Values (default) | Meaning |
| ----- | -------| ------- |
| data_type |  'syn' or 'ideal_time'  ('syn')| whether the matrices is defined as an ideal time delays matrices or ray-based (the generation code saves the measurements differently depending on the measurement type). If full pipeline use 'syn' too. 
|momentum | [0-1] | Initial momentum value |
|init_type | 'Lt', 'constant' ('Lt') | Initialization type. If 'constant' also feed 'c_init' parameters with the constant SoS value to use.
exp_name | Any string | Name of the folder to save models, summaries.
n_interp_knots_data | Integer| Number of knots to use for data term activation function
n_interp_knots_reg | Integer | Number of knots to use for regularization term activation function
D_standardize | Bool| Whether to standardize or not. If False it learns the mean.
weight_us| Bool  | Use undersampling rate dependent weighting of data term
readjust_rng| Bool | Use adaptive interpolator (otherwise fixed)
use_temperature | Bool | Whether to progressively decrease exp weighting. If False tau = 5.
decrease_lr | Bool  | Whether to progressively decrease learning rate.
n_iter | Integer | Number of training iterations
n_filters_vn | Int | Number of filters to use in the VN
filter_sz_vn | Int | Filter size in VN
n_layers | Int  | Number of unrolled iteration
use_spatial_filter_weighting | Bool | Whether to use spatial filtering
use_reg_activation_reg | Bool | Wheter to use regularization term activation function smoothing. Set lambda_reg accordingly.
use_reg_activation_data | Bool | Wheter to use data term activation function smoothing. Set lambda_data accordingly.
lambda_reg | Int | Value used in the regularization of the loss for the activation function smoothing.
csv | .csv file | File with the specifications of the input data for inference. 
KL | Bool | Whether to use Bayesian Variational Inference
alpha_KL | Int | Alpha parameter of Bayesian Variational Inference
beta_KL | Int | Beta parameter of Bayesian Variational Inference
aleat | Bool | Whather to compute the aleatoric uncertainty
drop | Bool | Whather to use MC dropout
rate | Int | Dropout rate magnitude for MC dropout, used if drop is True
n_samp | Int | Number of samples done at inference time, if plain VN it should remain as 1. If working with MC dropout or Bayesian Variational Inference 100 is recommended.
restore | Bool | Whether to start training from a previously saved model. Set cpkt to the checkpoint number to restore
filename | string | Name of the training set
 filename_val_fullpipeline | string | Name of the fullpipeline validation set |
 lr_init | float | Initial learning rate
 val_interv | int  | validate step freq
 print_interv | int | print summary statistics freq
 save_matrices | Bool | Whether to save training matrices for debugging. See save_matrices_interv accordingly.
 save_matrices_val | Bool |  Whether to save validation matrics matrices for debugging
 save_model_interv | Int | Save model frequency
 n_val | Int | Number of images from the training file to set apart for validation. 
 mix | Bool | Whether to mix two training file
 mix_triple | Bool | Whether to mix a third training file.
 filename_mix | string | Name of the second training file
 mix_type | | data_type of second training set
filename_mix_triple | string | Name of the third training file
mix_type_triple | | data_type of third training set
 p_mix | int | number of point to take from the second training set (per batch)
  p_mix_triple | int | number of point to take from the third training set (per batch)
  msz_x, msz_y | int | measurement size
  NA | int | Number of channels combinations, 6 for MS data, 15 for VS data
  n_batch | int | Batch size, generally 16 as default
  cost_type | 'recon_iter' or 'recon' | 'recon_iter' to use exp weighting. 
  noise_rate | [0-1] | Max noise rate
  usm_rate | [0-1] | Max undersampling rate
  random_mask_type | 'patchy' or 'plain' | Patchy or incoherent undersampling mask
minx_data, maxx_data, minx_reg, maxx_reg | | Min and max for activation function range in case fixed interpolator is used. Ignored if adaptive interpolation is used. 

### 4. Visualize the reconstructions and evaluate the Variational Network parameters.
###### This section corresponds to work done by Melanie Bernhardt
If studying the behaviour of the original VN network is desired, the codes can be found in `/evaluation/plotting_scripts`:
- To visualize the phantom reconstruction use `phantom_eval_vis.m`.
- To visualize the parameters of the trained model (such as filters, activations function etc.) use `vis_script_model.m`
- To visualize the different reconstruction experiment from our custom test set use `evaluation_vis_script.m`
- To investigate the domain shift use `investigate_diff_meas.m`
- To produce the evaluation plots of RMSE compared to lbfgs for one single model use `eval_vn_lbfgs.m`
- To visualize some details of the unrolled reconstruction use `vis_script_layers.m`. 
- To produce all evaluation plots from thesis run `eval_exp.m` (assumes all the models are saved in the same folder as described by the config file in this repository.)

### 5. Visualize reconstructions and evaluate the uncertainty estimation methods building on the Variational Network.
All the codes in this analysis are found in the folder `evaluation/analysis`:

- The Matlab file `Sonia_report.m`, visualizes the Virtual Source data, correlation maps, confidence maps and residuals used in the report.
- The Python file `rmse_globaluncertainty.py` studies the RMSE of all uncertainty methods for a desired intput file.
- The Jupyter Notebook `Report.ipynb` includes the graphs used in the report with the comparison of RMSE between different network architectures and uncertainty configurations. Together with examples of ΔSoS distributions and joint histograms for pixelwise uncertainty computation.
- The Jupyter Notebook `Evaluation_sim.ipynb` includes all the quantitative and qualitative analysis of simulated data using all three uncertainty estimation methods.
- The Jupyter Notebook `Evaluation_phantom.ipynb` includes all the quantitative and qualitative analysis of phantom data using all three uncertainty estimation methods.
### 6. Analyze the clinical data for the differential diagnosis case study
All the codes in this analysis are found in the folder `evaluation/analysis`:
- `Evaluation_clinic.ipynb` Jupyter Notebook that includes all the quantitative and qualitative analysis of the clinical data using all three uncertainty estimation methods. This same code has been replicated and run only for the corresponding methods for each case of uncertainty, and carcinoma or fibroadenoma for better visualization. The corresponding files with the results are found in: `Evaluation_clinic_CA_bay.ipynb`, `Evaluation_clinic_FA_bay.ipynb`, `Evaluation_clinic_CA_drop.ipynb`,`Evaluation_clinic_FA_drop.ipynb`,`Evaluation_clinic_CA_ens.ipynb`,`Evaluation_clinic_FA_ens.ipynb`
- `Stats_analysis.ipynb` This file carries out the statistical analysis on the differential diagnosis case study, generating the metrics for each frame selection method, the ΔSoS for each data pair and ROC curves for the classification problem. 
- `creating_LBFGS_data_Dieter.py` It generates the results on LBFGS obtained from Dieter Schweizer on Clinical data for baseline comparison
- `stats_analysis_lbfgs.ipynb` This notebook gives an overview of the different baselines based on LBFGS and different background selection and frame selection methods for the differential diagnosis study. 

### 7. Additional directories
The folder `results` includes the `.npy` results coming from ΔSoS of the clinical study and they are described throughout the codes when used/needed. 
The folder `runs` includes the final checkpoints and results of the final models used in this project and the LBFGS reconstruction results. The file `README_runs.md` in such directory gives a detailed explanation of all folders. 
The `train_vn.sh` and `test_vn.sh` are examples of shell files that can be used to run the VN at the Computer Vision Laboratory at ETH Zurich. 