function opts = Fukuda_opts(opts,varargin)


try
    metafile = [opts.general.datapath 'Meta_' opts.general.filename(1:end-3) 'm'];
    run(metafile);
    opts.acq.Transducer.probe = mSignalProcessingChain.mAssociatedUsModeSettings.mTransducerLabel;
catch
    disp('No meta file found. Using default settings.')
end
% defining the defaults of the Fukuda acquisition

if ~isfield(opts,'acq') ||  ~isfield(opts.acq,'Transducer') || ~isfield(opts.acq.Transducer,'probe') 
    opts.acq.Transducer.probe = 'FUT-LA385-12P'; % for FUT-LA385-12P Transducer
end
%     opts.acq.Transducer.probe = 'BPL55(5-9)'; % for BPL55 5-9 Transducer

opts.version = '1.03';
opts.device = 'Fukuda';
opts.acq.Transducer.channels = 128;
opts.acq.RX.fs = 40.96e6;
opts.acq.TX.fs = 122.88e6;
if nargin > 1
    RF = varargin{1};
    opts.acq.RX.samples = size(RF,1);
end

if strcmpi(opts.acq.Transducer.probe,'FUT-LA385-12P') || strcmpi(opts.acq.Transducer.probe,'LA38')
    opts.acq.Transducer.pitch = 300E-6; % for FUT-LA385-12P Transducer
    disp('FUT-LA385-12P Transducer selected')
elseif strcmpi(opts.acq.Transducer.probe,'BPL55') || strcmpi(opts.acq.Transducer.probe,'BPL55(5-9)')
    opts.acq.Transducer.pitch = 430E-6; % for BPL55(5-9) Transducer
    disp('BPL55(5-9) Transducer selected')
end
opts.acq.RX.depth_start = 0;
width = opts.acq.Transducer.channels*opts.acq.Transducer.pitch;
try 
    maxdist = opts.acq.RX.samples/opts.acq.RX.fs*1540/2;
    opts.acq.RX.depth = sqrt(maxdist^2 - width^2);
end
opts.acq.TX.TXfreq = 5e6;
opts.acq.TX.PulseLengthHalf = 4;

end