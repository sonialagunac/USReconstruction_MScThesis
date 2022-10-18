# Datasets description
##### MSc Thesis - Sonia Laguna - ETH Zurich
This file includes the description of the data found in the folder `data_original` and used in this project. 
The directories found are the following: 
- `Test`: Includes the test synthetic Multistatic data used for the Variational network: Fullpipeline, Ray-based and the Ray-Based test set of VS translated into MS data. 
- `Validation`: Multistatic synthetic data used for validation of the network: Fullpipeline and Ray-Based
- `Train`: Multistatic synthetic data used to train the network: Fullpipeline and Ray-Based
- `Phantom`: Phantom multistic data including masks and the phantom's in previous work (Melanie's).
- `Phantom_mert`: Phantom multistic data including masks and the phantom's in previous work (Mert's).
- `MS_clinical`: Multistatic clinical data studied in this project ordered by lesion with their masks. Includes directories with data from the mpBUS study for each corresponding subject. 
- `MS_IC_experiment`: Includes the test data used in the experiments with varying contrast in this thesis. 
- `VS`: Includes all the Virtual Source data used. The single files include the test, training and validation data used in this study with fullpipeline and ray-based cases, together with the L matrix. `subjects` has all the data coming from the clinical study. `subjects\CA_subjects` and `subjects\CA_subjects` include the relevant data for the subjects under study form the mpBUS study and created per subjects. `subjects\mat_sos` has all the data used as input in the VN with the measurements and masks. `Dieter_qualitative_results` includes the qualitative LBFGS reconstructions carried out by Dieter. 
