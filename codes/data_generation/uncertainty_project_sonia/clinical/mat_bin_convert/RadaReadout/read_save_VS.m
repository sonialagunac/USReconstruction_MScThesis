% Sonia Laguna - ETH Zurich MSc Thesis
% Adapted from Dieter Schweizer
% Reading bin files into mat files for clinical data VS
clear
close all
clc
addpath(genpath('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation_Sonia/mat_bin_clinical/RadaRedout'));
default_data_path = '..\';

sub_1 = 'mpBUS019/'
sub_2 = dir (['/scratch_net/biwidl307/sonia/data_original/VS/subjects/CA_subjects',sub_1])
for ind = [3: length(sub_2)]
     name = sub_2(ind).name;
     sub = [sub_1, name, '/']; 
    data_path = ['/scratch_net/biwidl307/sonia/data_original/VS/subjects/CA_subjects' , sub];
    if length(dir(data_path))<=2
        continue
    end
    data_file = dir ([data_path '*.bin*']).name;

    opts.device = 'Fukuda';
    opts.general.filename = data_file;
    opts.general.datapath = data_path;
    opts.postprocess.general.c = 1540;        % [m/s], initial speed of Sound for beamforming
    [RF, BmodeRF, opts] = Genreal_DataReadout(data_path,data_file,opts);
    if isempty(RF)
    error('Specified file does not contain interleaved planewave frames')
    end
    
    save([data_path, data_file(1:end - 3), 'mat'],'RF', 'BmodeRF', 'opts')
end