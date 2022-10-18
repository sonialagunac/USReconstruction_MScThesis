function opts = defaultPostprocessOpts(opts,varargin)
%% Copyright Richard Rau @ ETH Zurich, July 2019, rrau@ee.ethz.ch
% to inizialize the opts with defaults parameters

disp('Loading default parameters into the opts.postprocess structure - see defaultPostprocessOpts function for details')

%% todo: adapt such that if the value is already given it will not be overwritten.
% Define beamforming/sos recon axis and initialize sos map for beamforming
pixelsize_sos = [  2,   2].*opts.acq.Transducer.pitch; % ax/lat resolution for the sos recon
%pixelsize_bf =  [1/8, 1].*opts.acq.Transducer.pitch; % ax/lat resolution for the bf/dispTrack
pixelsize_bf =  [1/8, 1/2].*opts.acq.Transducer.pitch; % ax/lat resolution for the bf/dispTrack
if isfield(opts.acq.RX,'depth')
    Depth = opts.acq.RX.depth;% fix(1e4*opts.acq.RX.samples/opts.acq.RX.fs*c_initial/2)/1e4; % Define up to which depth the reconstruction should be carried out
else
    Depth = fix(size(RF,1)/opts.acq.RX.fs*1500/2);
end
%Depth = 0.05;
Width = (opts.acq.Transducer.channels-1)/2*opts.acq.Transducer.pitch;
xax_sos = [-Width : pixelsize_sos(2) : Width+0.00001];  xax_sos = xax_sos-mean(xax_sos);
zax_sos = [0e-3:pixelsize_sos(1):Depth];
%zax_sos = [0e-3:pixelsize_sos(1):0.04];
xax_bf = xax_sos(1):pixelsize_bf(2):xax_sos(end); 
zax_bf = [zax_sos(1):pixelsize_bf(1):zax_sos(end)];

% general parameters
opts.postprocess.general.save = 0; % intermediate steps are not saved
opts.postprocess.general.plot = 0; % not plotting each processing step
opts.postprocess.general.DistancePrecomp_Path = ['/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation/Precomp_Distance_Matrix/'];% This path is used to precompute the distance matrices to speed up computations (some precomputed matricex can be found here: https://www.dropbox.com/sh/v4yu6e0d4b2dzeu/AABnHtNpJ5bmC5p-4iVizfhEa?dl=0)
if ~isfield(opts.postprocess.general,'processingUnit')
    opts.postprocess.general.processingUnit = 'gpu'; % select here gpu if pc is equipped with one
end
if ~isfield(opts.postprocess.general,'c')
    opts.postprocess.general.c = 1500;
end

% for resampling of the data
opts.postprocess.ResampleRF.samplesPerWavelength = 32; % e.g. set 32 for 32 samples per wavelength. 32 is good!

if ~isfield(opts.postprocess.general,'varsavepath')
    opts.postprocess.general.varsavepath = [];
end

% beamforming
%opts.postprocess.BF.method = 'DASwithSOSmap_psf_aligned';
opts.postprocess.BF.method = 'DASwithSOSmap'
opts.postprocess.BF.x_axis = xax_bf; % define the axis of the beamforming grid
opts.postprocess.BF.z_axis = zax_bf; % define the axis of the beamforming grid
opts.postprocess.BF.sos_x_axis = xax_sos; % this sos map is used for beamforming
opts.postprocess.BF.sos_z_axis = zax_sos; % this sos map is used for beamforming 
opts.postprocess.BF.TXchannels = [1:opts.acq.Transducer.channels]'; % TX channels to be beamformed
opts.postprocess.BF.maxRXchannels = opts.acq.Transducer.channels;
opts.postprocess.BF.f_number_TX = .5;
opts.postprocess.BF.f_number_RX = 1;
opts.postprocess.BF.sos_data = opts.postprocess.general.c .* ones(numel(zax_sos),numel(xax_sos)); % this is the initial sos map for used for the first iteration
opts.postprocess.BF.psf_angles = [-5,0,5];
if isfield(opts,'sim')
   % opts.postprocess.BF.offset = 275e-9; %opts.acq.TX.PulseLengthHalf/2/opts.acq.TX.TXfreq/2;
   opts.postprocess.BF.offset = 0; 
elseif strcmpi(opts.acq.Transducer.probe,'FUT-LA385-12P')
    opts.postprocess.BF.offset = -1e-6; 
else
    if ~isfield(opts.postprocess.BF,'offset')
        warning('RR: offset for BF not calibrated for this probe')
        opts.postprocess.BF.offset = -2.0e-6; % in s, extra delay for beamforming (has to be changed depending on acqusition system, for k-wave and fukuda = 0 seems good)
    end
end

% Displacement Tracking Parameters
opts.postprocess.DispTrack.method = 'NCC_psf_aligned';
%opts.postprocess.DispTrack.method = 'NCC'
%opts.postprocess.DispTrack.threshhold = 0.2; % thresholds the displacements if correlation is not good, 0.2 seems good
opts.postprocess.DispTrack.x_axis = xax_bf; % define the axis of the beamforming grid
opts.postprocess.DispTrack.z_axis = zax_bf; % define the axis of the beamforming grid

% sos reconstruciton parameters
opts.postprocess.sos_minus.Solver = 'lbfgs';
opts.postprocess.sos_minus.optimization_c1 = 1e-4; % for the gradient in lbfgs optimization algorithm
opts.postprocess.sos_minus.optimization_c2 = 1e-4; % for the gradient in lbfgs optimization algorithm
opts.postprocess.sos_minus.optimization_max_iters_lbfgs = 5000; % for the lbfgs optimization algorithm
opts.postprocess.sos_minus.smoothing = .5e-3; % blurring the reconstructed sos map by this sigma value to avoid artifacts (in m), usually set to ~.5e-3
opts.postprocess.sos_minus.x_axis = xax_sos; % define the axis of the sos recon grid
opts.postprocess.sos_minus.z_axis = zax_sos; % define the axis of the sos recon grid

if nargin > 1
    METHOD = varargin{1};
    opts.postprocess.BF.input = METHOD; % if PW or MS data should be beamformed, change METHOD above
    opts.postprocess.sos_minus.input = METHOD; %'MS' or 'PW' for multistatic based sos reconstruction of plane wave based reconstruction. Has to be adapted according to the data input to this processing block, change this above
end




end