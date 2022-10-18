function RF = bandpassRF(RF,opts,bw,ord)
%% Copyright Richard Rau @ ETH Zurich, July 2019, rrau@ee.ethz.ch
% bandpass filters the RF data 
if nargin < 4
    ord = 9;
end

if nargin < 3
    bw = .6;
end
[b2,a2] = butter(ord,(opts.acq.TX.TXfreq*(1-bw))/(opts.acq.RX.fs/2),'high');
[b,a] =   butter(ord,(opts.acq.TX.TXfreq*(1+bw))/(opts.acq.RX.fs/2));
cl = class(RF);
RF = filtfilt(b,a,double(RF));
RF = filtfilt(b2,a2,double(RF));
if strcmpi(cl,'single')
    RF = single(RF);
end

end