function [rawbeam_data, rawbeam_RXegTX, bmode_data] = Fukuda_bin2rawbeams(in_data, TXcoding)
% function [rawbeam_data, rawbeam_RXegTX] = RADA_bin2rawbeams(in_data, TXcoding)
% Dieter Schweizer
% Sept. 2018, CAiM Group, ETH Zürich
%
% The data are read from the 32-bit binary input and returned as an array of
% raw data beams (rawbeam_data) for each receive channel (always 128 RX channels) up to
% the number of TX sequences ocurring in the data set.
%
% A second data array (rawbeam_RXegTX) is given back which contains the RX beams which are
% located at the single element TX position.
%
% The data are input are encoded according to TXcoding:
%       1:  uncoded single element multi-static data
%       2:  128 element bi-polar Hadamard encoding
%       3:  128 element uni-polar Hadamard H+ only encoding
%       4:  128 element planewave
%       6:  128 element pseudo MST version C

%% Local constants
NUM_RECEIVECHANNELS = 128;  % Number of receive channels on UF-760AG
DISPLAY_SUMMARY = 0;        % 1 = Display summary of detected raw aquisition cycles
DISPLAY_DATA = 0;           % 1 = Display MST and RXegTX data as absolute values
SAVE_MATFILE = 0;           % 1 = save workspace as .mat file
FULL_DATA = 0;              % 1= ignore Start sample information from header and read 1000 samples beyond the stop sample
FAST = 1;                   % 0 = Detailed data parsing, 1 = Fast parsing (with minimum steps)

%%
%% We need to detect if a header can be found
found = 0; flag = 0;
ii = 1;
while ~found
    bit = bitget(in_data(ii),2);
    if flag && ~bit
        firstind = ii;
        found = 1;
    elseif ~flag && bit
        flag = 1;
    elseif ii > length(in_data)
        error('Header not found');
    end
    ii = ii+1;
end

if FAST == 1
    ind=find(diff(bitand(in_data,2))<0)+1; % find indices of headers
    % extract headers
    header=uint16(bitshift([in_data(ind) in_data(ind+1) in_data(ind+2) in_data(ind+3) in_data(ind+4) in_data(ind+5) in_data(ind+6) in_data(ind+7)],-16));

    %% extract required header information
    head.position=ind;
    head.pw_packet_nr=header(:,1+1);
    head.pw_beg_nr=header(:,1+2);
    head.pw_end_nr=header(:,1+3);
    head.pw_scan_type=bitand(header(:,1+4),15);

    if FULL_DATA ==1
        head.pw_beg_nr(:)=0;
        head.pw_end_nr=header(:,1+3)+1000;
    end

    %% Analyze raw data file
    % Assuming that the data contain first the raw data and then the B mode
    % data then 'stop' is the last beam of transport before switching to B mode
    start = 1;
    stop = find((head.pw_packet_nr(2:end)==0) & ((head.pw_scan_type(2:end)==0)| (head.pw_scan_type(2:end)==14)) );

    rawbeam_ind = find(head.pw_scan_type(start:stop)==14);
    RAW.AquiCycles = max(head.pw_packet_nr(rawbeam_ind))-min(head.pw_packet_nr(rawbeam_ind)) + 1;
    RAW.samples  = double(head.pw_end_nr(2)-head.pw_beg_nr(2)+1);

    %% Map raw data to output data array
    rawbeam_data   = zeros(RAW.samples,NUM_RECEIVECHANNELS,RAW.AquiCycles/2); %(samples,RX,TX)

    rx_counter = 0;
    tx_counter = 0;

    for i=start:stop

        switch (head.pw_scan_type(i))
            case {15} % data storage
                indices = head.position(i)+6+2+4*(double(head.pw_beg_nr(i)+1):double(head.pw_end_nr(i)+1));

                rawbeam_data(:, 0 + (rx_counter+1),tx_counter+1)=double(in_data(indices));
                rawbeam_data(:, 1 + (rx_counter+1),tx_counter+1)=double(in_data(indices+1));
                rawbeam_data(:, 2 + (rx_counter+1),tx_counter+1)=double(in_data(indices+2));
                rawbeam_data(:, 3 + (rx_counter+1),tx_counter+1)=double(in_data(indices+3));

                %post update
                if(((rx_counter+4))>=NUM_RECEIVECHANNELS)                
                    tx_counter = tx_counter + 1; 
                    rx_counter = 0;
                else
                    rx_counter = rx_counter + 4;
                end

        end

    end
    
    %% Search for B mode frame
    bmode_ind = find(head.pw_scan_type==0);
    start = bmode_ind(1);
    if (head.pw_scan_type(bmode_ind(1)+128)==0)
        stop = bmode_ind(192);
    else
        stop = bmode_ind(128);
    end
    BMode.AquiCycles = max(head.pw_packet_nr(bmode_ind))-min(head.pw_packet_nr(bmode_ind)) + 1;
    BMode.samples  = double(head.pw_end_nr(bmode_ind(1))-head.pw_beg_nr(bmode_ind(1))+1);

    %% Map raw data to output data array
    bmode_data   = zeros(BMode.samples,BMode.AquiCycles*2); %(samples,RX)
    rx_counter = 0;

    for i=start:stop

        indices = head.position(i)+6+2+4*(double(head.pw_beg_nr(i)+1):double(head.pw_end_nr(i)+1));

        bmode_data(:, 0 + (rx_counter+1))=double(in_data(indices));
        bmode_data(:, 1 + (rx_counter+1))=double(in_data(indices+1));

        rx_counter = rx_counter + 2;
    end
    if DISPLAY_SUMMARY == 1
        fprintf('Bmode Frame : \n');
        BMode
    end
  
else
    %% We need to check whether we can use the cached indices
    try
        load([mfilename(pwd, '/DataCache/cache.mat')]);
    catch
        cache = struct('dlength',0,'ind',[]);
    end

    if cache.dlength == length(in_data) && cache.ind(1) == firstind
        ind = cache.ind;
    else
        ind=find(diff(bitand(in_data,2))<0)+1;
        cache.ind = ind;
        cache.dlength = length(in_data);
        cache.d1 = in_data(1);
        mkdir([pwd '/DataCache']);
        save([pwd '/DataCache/cache.mat'],'cache');
    end

    %% remove Raw data bit (MSB) from header 1
    in_data(ind) = bitand(in_data(ind),2147438647); % mask MSB of 32 bit data

    header=uint16(bitshift([in_data(ind) in_data(ind+1) in_data(ind+2) in_data(ind+3) in_data(ind+4) in_data(ind+5) in_data(ind+6) in_data(ind+7)],-16));

    %% extract header information
    head.position=ind;

    subch=bitand(bitshift(header(:,1),-8),15);           % extract quad/dual/single h7
    head.subchan= 1*(subch==1)+2*(subch==3)+4*(subch==15);
    head.pw_scan_id=bitand(header(:,1), 15);
    head.pw_sync_field=bitand(bitshift(header(:,1),-4),15);
    head.pw_rxi=bitand(bitshift(header(:,1),-8),15);
    head.pw_packet_nr=header(:,1+1);
    head.pw_beg_nr=header(:,1+2);
    head.pw_end_nr=header(:,1+3);
    head.pw_scan_type=bitand(header(:,1+4),15);
    head.pw_beam_per_frame=bitand(header(:,1+5),1023);

    if FULL_DATA ==1
        head.pw_beg_nr(:)=0;
        head.pw_end_nr=header(:,1+3)+1000;
    end

    %% Analyze raw data file
    % Assuming that the data contain first the raw data and then the B mode
    % data then 'stop' is the last beam of transport before switching to B mode
    start = 1;
    stop = find((head.pw_packet_nr(2:end)==0) & ((head.pw_scan_type(2:end)==0)| (head.pw_scan_type(2:end)==14)) );

    rawbeam_ind = find(head.pw_scan_type(start:stop)==14);
    if not(isempty(rawbeam_ind))
        RAW.AquiCycles = max(head.pw_packet_nr(rawbeam_ind))-min(head.pw_packet_nr(rawbeam_ind)) + 1;
        % packet number varies from 0 to 15
        % 1 acquistion packet (type 14) is followed by 16 transfer packets (type 15)

        rawbeam_ind = find(head.pw_scan_type(start:stop)==15);

        RAW.BeamsPerAcqui = (max(head.pw_packet_nr(rawbeam_ind))-min(head.pw_packet_nr(rawbeam_ind)) + 1)*4;

        RAW.samples  = double(head.pw_end_nr(rawbeam_ind(1))-head.pw_beg_nr(rawbeam_ind(1))+1);
        RAW.frames   = double(length(find((head.pw_scan_type(start:stop)==14) & (head.pw_packet_nr(start:stop)==0))));
        RAW.totalbeams = length(rawbeam_ind)*4;

    else
        error('No RawData found');   
    end

    %% Map raw data to output data array
    rawbeam_data   = zeros(RAW.samples,NUM_RECEIVECHANNELS,RAW.AquiCycles/2); %(samples,RX,TX)

    rx_counter = 0;
    tx_counter = 0;

    for i=start:stop

        switch (head.pw_scan_type(i))
            case {15} % data storage
                indices = head.position(i)+6+2+4*(double(head.pw_beg_nr(i)+1):double(head.pw_end_nr(i)+1));

                rawbeam_data(:, 0 + (rx_counter+1),tx_counter+1)=double(in_data(indices));
                rawbeam_data(:, 1 + (rx_counter+1),tx_counter+1)=double(in_data(indices+1));
                rawbeam_data(:, 2 + (rx_counter+1),tx_counter+1)=double(in_data(indices+2));
                rawbeam_data(:, 3 + (rx_counter+1),tx_counter+1)=double(in_data(indices+3));

                %post update
                if(((rx_counter+4))>=NUM_RECEIVECHANNELS)                
                    tx_counter = tx_counter + 1; 
                    rx_counter = 0;
                else
                    rx_counter = rx_counter + 4;
                end

        end

    end
end

%% Report summary of raw data reading
if DISPLAY_SUMMARY == 1, RAW, end

%% Decoding according to TXcoding
switch TXcoding
    case 1 % Uncoded single element MST data

    case 2 % 128 element bi-polar Hadamard encoding
        rawbeam_128_predecode = zeros(size(rawbeam_data,1),size(rawbeam_data,2),128);
        for i = 1:64
          rawbeam_128_predecode(:,:,i) = rawbeam_data(:,:,2*i-1) + rawbeam_data(:,:,2*i);
        end
        for i = 1:64
            rawbeam_128_predecode(:,:,i+64) = rawbeam_data(:,:,2*i-1) + rawbeam_data(:,:,i+128);
        end
        paramsH = struct('order', 128, 'Nel', 128, 'codeType', 1);
        rawbeam_128_decode = CAiM_WalshHadamard_Decode(rawbeam_128_predecode, paramsH);
        rawbeam_data = rawbeam_128_decode;
    case 3 % 128 element uni-polar Hadamard H+ only encoding
        rawbeam_128_predecode = zeros(size(rawbeam_data,1),size(rawbeam_data,2),128);
        for i = 1:64
          rawbeam_128_predecode(:,:,i) = rawbeam_data(:,:,2*i-1) + rawbeam_data(:,:,2*i);
        end
        for i = 1:64
            rawbeam_128_predecode(:,:,i+64) = rawbeam_data(:,:,2*i-1) + rawbeam_data(:,:,i+128);
        end
        paramsH = struct('order', 128, 'Nel', 128, 'codeType', 3);
        rawbeam_128_decode = CAiM_WalshHadamard_Decode(rawbeam_128_predecode, paramsH);
        rawbeam_data = rawbeam_128_decode;
    case 4 % 128 element Planewave
        % Sum-up the rx-data of 2 following acquisition cycles to get a 128 TX - 128
        % RX plane wave dataset
        NbOfPlanes = RAW.AquiCycles/4;
        rawbeam_128_sum = zeros(size(rawbeam_data,1),size(rawbeam_data,2),NbOfPlanes);
        for i = 1:NbOfPlanes
          rawbeam_128_sum(:,:,i) = rawbeam_data(:,:,2*i-1) + rawbeam_data(:,:,2*i);
        end
        rawbeam_data = rawbeam_128_sum;
    case 6 % 128 element pseudo MST version C
        rawbeam_sum = zeros(size(rawbeam_data,1),size(rawbeam_data,2),128);
        for i = 1:64
            rawbeam_sum(:,:,2*i-1) = (rawbeam_data(:,:,2*i-1) + rawbeam_data(:,:,2*i))/2;
        end
        for i = 1:64
            rawbeam_sum(:,:,2*i) = (rawbeam_data(:,:,2*i-1) - rawbeam_data(:,:,2*i))/2;
        end
        rawbeam_data = rawbeam_sum; 
    case 7 % 128 element uni-polar Hadamard H+ only encoding with 256 TX cycles
        rawbeam_128_predecode = zeros(size(rawbeam_data,1),size(rawbeam_data,2),128);
        for i = 1:128
          rawbeam_128_predecode(:,:,i) = rawbeam_data(:,:,2*i-1) + rawbeam_data(:,:,2*i);
        end
%         for i = 1:64
%             rawbeam_128_predecode(:,:,i+64) = rawbeam_data(:,:,2*i-1) + rawbeam_data(:,:,i+128);
%         end
        paramsH = struct('order', 128, 'Nel', 128, 'codeType', 3);
        rawbeam_128_decode = CAiM_WalshHadamard_Decode(rawbeam_128_predecode, paramsH);
        rawbeam_data = rawbeam_128_decode;
end

if FAST == 1
    rawbeam_RXegTX = [];
    return;
else
    %% Generate RXegTX data array and MST data array (used for display only)
    rawbeam_RXegTX = zeros(RAW.samples,NUM_RECEIVECHANNELS);

    if TXcoding ~= 4
        rawbeam_RXegTX(:,1) = rawbeam_data(:,1,1);
        rawbeam_MST = rawbeam_data(:,:,1);

        for i = 2:NUM_RECEIVECHANNELS
            rawbeam_RXegTX(:,i) = rawbeam_data(:,i,i);
            rawbeam_MST = [rawbeam_MST rawbeam_data(:,:,i)];
        end
    else
        zeroDegreeAcqu = int16(size(rawbeam_data,3)/2);
        rawbeam_RXegTX = rawbeam_data(:,:,zeroDegreeAcqu);

        rawbeam_MST = rawbeam_data(:,:,1);
        for i = 2:NbOfPlanes
            rawbeam_MST = [rawbeam_MST rawbeam_data(:,:,i)];    
        end
    end


    %% Plotting RAW Data %%%%%%%%%%%%%%%%%%%%%%%%
    if DISPLAY_DATA == 1
        figure('Name', 'RawData Plots', 'Position', [1450, 200, 400, 900])
        % All TxRx pairs plot
            subplot(3,1,1);
            data_mst=bitshift(int32(rawbeam_MST),-14);
            plot_data_mst=abs(hilbert(data_mst));
            image(plot_data_mst);    
            title('Multi-static-data: all 128 x 128 TXRX pairs');
            xlabel('Pair index');
            ylabel('Sample');

        % TXegRX pair plot
            subplot(3,1,2);
            data_RXegTX=bitshift(int32(rawbeam_RXegTX),-14);
            plot_data_RXegTX=abs(hilbert(data_RXegTX));
            image(plot_data_RXegTX);
            title('Multi-static-data: only TXeqRX pairs');
            xlabel('Pair index');
            ylabel('Sample');

         % TXegRX = 64 mst-signal plot
            subplot(3,1,3);      
            plot(rawbeam_RXegTX(:,64));
            hold on
            % envelope
            plot(abs(hilbert(rawbeam_RXegTX(:,64))),'r-');  grid on;
            title('Multi-static-data TXegRX pair = 64');
            xlabel('Sample');
            ylabel('Signal magnitude');
    end

    %% Saving workspace
    if (SAVE_MATFILE==1)
         uisave;
    end
end

end

