%% Copyright richard Rau, Jul 2019 @ ETH Zurich
% This fle reads out Fukuda Datasets

function [RF, BMode, opts] = FukudaReadout(pathname,file,opts)



if ~isfield(opts,'acq') || ~isfield(opts.acq,'sequence') || ~isfield(opts.acq.sequence,'method') 
    % try to grab sequence from filename
    if contains(file,'MS')
        opts.acq.sequence.method = 'Multistatic';
    elseif contains(file,'VS')
        opts.acq.sequence.method = 'VS';
    elseif contains(file,'WH')
        opts.acq.sequence.method = 'Hadamard';
    elseif contains(file,'PMSCT')
        opts.acq.sequence.method = 'PseudoMST';
    elseif contains(file,'PW')
        opts.acq.sequence.method = 'PlaneWave';
    else
        choice = menu(file,'Multistatic','Hadamard','PlaneWave','Pseudo MST','Virtual Source','Hadamard H+, 256tx');
        switch choice
            case 1, opts.acq.sequence.method = 'Multistatic';
            case 2, opts.acq.sequence.method = 'Hadamard';
            case 3, opts.acq.sequence.method = 'PlaneWave';
            case 4, opts.acq.sequence.method = 'PseudoMST';
            case 5, opts.acq.sequence.method = 'VS';
            case 6, opts.acq.sequence.method = 'Hadamard H+, 256tx';
            case 0, return
        end
    end
    disp(['Dataset sequence: ' opts.acq.sequence.method])
end

if strcmpi(opts.acq.sequence.method,'multistatic') || strcmpi(opts.acq.sequence.method,'Multistatic') || strcmpi(opts.acq.sequence.method,'MS') || strcmpi(opts.acq.sequence.method,'MST')
    TXcoding = 1;
elseif strcmpi(opts.acq.sequence.method,'VS')
    TXcoding = 1;
elseif strcmpi(opts.acq.sequence.method,'PW') || strcmpi(opts.acq.sequence.method,'PlaneWave')
    TXcoding = 4;
elseif strcmpi(opts.acq.sequence.method,'WH') || strcmpi(opts.acq.sequence.method,'Hadamard') 
    TXcoding = 2;
elseif strcmpi(opts.acq.sequence.method,'Hadamard H+, 256tx') 
    TXcoding = 7;
elseif strcmpi(opts.acq.sequence.method,'PseudoMST') || strcmpi(opts.acq.sequence.method,'PMST') || strcmpi(opts.acq.sequence.method,'PMS') || strcmpi(opts.acq.sequence.method,'PM') 
    TXcoding = 6;
else
    error('Unknown sequence, please define in opts.acq.sequence.method')
end

fid = fopen([pathname, file]);
raw_data = fread(fid,inf,'int32=>int32');
fclose(fid);  
[RF, ~, BMode] = Fukuda_bin2rawbeams(raw_data, TXcoding);

opts = Fukuda_opts(opts,RF);

if strcmpi(opts.acq.sequence.method,'PlaneWave') % define opts for PW
    load([pathname, filesep, 'SequenceParameters_' file(1:end-4) '.mat']);
    opts.postprocess.PWG.steering_angle = PWangles;
    opts.postprocess.PWG.delay = Delays' * 122.88e6 / opts.acq.RX.fs;
    opts.postprocess.PWG.apod = (Apodization' + 1) .* 0.125;
    opts.postprocess.general.c = c0_tissue;
end

if strcmpi(opts.acq.sequence.method,'VS') % define opts for VS
    load([pathname, filesep, 'SequenceParameters_' file(1:end-4) '.mat']);
    if size(TxCenter,1) ~= size(RF,3)
        error(sprintf('RR: Sequence Parameter file predicts %2.0f TX events, but RF data has %2.0f TX events',size(TxCenter,1),size(RF,3)))
    end
    VS_idx = find(TxCenter(:,4)==0);
    PW_idx = find(TxCenter(:,4)==1); 
    RF_n = RF; 
    clear RF;
    RF.VS = RF_n(:,:,VS_idx);
    RF.PW = RF_n(:,:,PW_idx);
    %%
    VS_Aperture_unique = unique(diff(TxCenter(VS_idx,2:3)'))+1;
    if numel(VS_Aperture_unique) > 1
        fprintf('Apertures for VS:')
        for n = 1:numel(VS_Aperture_unique)
            fprintf(' %2.0f ',VS_Aperture_unique(n))
        end
        fprintf('\n')        
        error('RR: More than one aperture for VS. Postprocessing (BF, sos_recon) currently only allows single aperture.')
    end
    if round(VS_Aperture_unique/2) == VS_Aperture_unique/2
        error('RR: Got an even aperture number, should be odd')
    end
    VS_apod = Apodization(VS_idx,:);
    if any(nnz(diff(VS_apod)))
        error('RR: Apodizations for VS seem to be varying. Postprocessing (BF, sos_recon) currently only allows single apodization')
    end
    VS_apod = VS_apod(1,:); % becaue they are all the same
%     if nnz(VS_apod) > VS_Aperture_unique
%         error('RR: Number of elements in apodization are larger than aperture.')
%     elseif nnz(VS_apod) < VS_Aperture_unique
%         warning(sprintf('RR: Number of elements in apodization (%2.0f) are smaller than aperture (%2.0f)',nnz(VS_apod),VS_Aperture_unique))
%     end    
%    VS_apod_center = sum([1:numel(VS_apod)].*VS_apod)/sum(VS_apod); % center of mass
    VS_apod_center = 33; % center of mass for odd apertures
    ap_half = (VS_Aperture_unique-1)/2;
    VS_apod = VS_apod(VS_apod_center+[-ap_half:ap_half]); % drop the zeros
    
    VS_TxCenter = TxCenter(VS_idx,1);
    elpos = ElementPositions1d(opts);
    VS_TxCenter_m = elpos(VS_TxCenter);
    
    
    if FastSequence
        chan_comb = [1:size(RF.VS,3)];
        chan_comb2 = chan_comb(2:end);
        chan_comb(end) = [];
        chan_comb = [chan_comb; chan_comb2]';
        wraparoundIdx = find(abs(diff(TxCenter(VS_idx,1))) > Deltachan);
        chan_comb(wraparoundIdx,:) = [];
        opts.acq.sequence.chan_comb = chan_comb;
    else
        chan_comb = [1:size(RF.VS,3)];
        chan_comb = reshape(chan_comb,2,[])';
        opts.acq.sequence.chan_comb = chan_comb;
    end
    
    opts.acq.sequence.c = c0_tissue;
    opts.acq.sequence.focus_ax = -TXfocus;
    opts.acq.sequence.focus_lat = VS_TxCenter_m;
    opts.acq.sequence.steering_angle = 0; % apadt this if steering is implementd in VS, currently not foreseen to be required
    opts.acq.RX.aperture = opts.acq.Transducer.channels; % adapt this if less than the full RX aperture is used
    opts.acq.TX.aperture = VS_Aperture_unique;
    opts.acq.TX.apod = zeros(opts.acq.Transducer.channels,nnz(VS_idx));
    opts.acq.TX.delay = zeros(opts.acq.Transducer.channels,nnz(VS_idx));
    for n = 1:nnz(VS_idx)
        idx = VS_idx(n);
        txcenter = TxCenter(idx,1);
        delay = nan(opts.acq.Transducer.channels,1);
        delay(txcenter+[-ap_half:ap_half]) = Delays(idx,VS_apod_center+[-ap_half:ap_half]);
        opts.acq.TX.delay(:,n) = delay/opts.acq.RX.fs/3;
        opts.acq.TX.apod(txcenter+[-ap_half:ap_half],n) = VS_apod;
    end
    
    if ~isempty(PW_idx)
        opts.acq.sequence.pw.steering_angle = PWAngles;
        PW_apod = Apodization(PW_idx(1),:);   
        PW_apod_center = sum([1:numel(PW_apod)].*PW_apod)/sum(PW_apod); % center of mass
        ap_half = numel(PW_apod)/2;
        for n = 1:nnz(PW_idx)
            idx = PW_idx(n);
            txcenter = TxCenter(idx,1);
            delay = nan(opts.acq.Transducer.channels,1);
            delay(txcenter-.5+[-ap_half+1:ap_half]) = Delays(idx,PW_apod_center-.5+[-ap_half+1:ap_half]);
            opts.acq.TX.PWdelay(:,n) = delay;
            opts.acq.TX.PWapod(txcenter-.5+[-ap_half+1:ap_half],n) = PW_apod;
        end      
    end
    disp(sprintf('VS:\nsamples:  %2.0f \nchannels:  %2.0f \nnTX:        %2.0f \nframes:     %2.0f  ' ,size(RF.VS,1),size(RF.VS,2),size(RF.VS,3),size(RF.VS,4)));
    disp(sprintf('PW:\nsamples:  %2.0f \nchannels:  %2.0f \nnTX:        %2.0f \nframes:     %2.0f  ' ,size(RF.PW,1),size(RF.PW,2),size(RF.PW,3),size(RF.PW,4)));
else
    disp(sprintf('samples:  %2.0f \nchannels:  %2.0f \nnTX:        %2.0f \nframes:     %2.0f  ' ,size(RF,1),size(RF,2),size(RF,3),size(RF,4)));
end
end