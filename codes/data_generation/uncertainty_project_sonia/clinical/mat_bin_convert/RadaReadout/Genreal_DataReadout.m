%% Copyright Richard Rau, Nov 2018 @ ETH Zurich
% This function is the framework to read out different datasets
% Example 1:
% [RF opts] = Genreal_DataReadout(DataPath,DataFile,opts) 
% read out DataFile from DataPata with opts structure given (usefule if opts.device is set). 
% Example 2:
% [RF opts] = Genreal_DataReadout(DataPath,DataFile) 
% reads out DataFile from DataPata. Question for Acqusition Device is asked. 
% Example 3:
% [RF opts] = Genreal_DataReadout(DataPath) 
% open a dialog in DataPata to select a file to readout
% Example 4:
% [RF opts] = Genreal_DataReadout() 
% open a dialog in pwd to select a file to readout

function [RF, Bmode, opts] = Genreal_DataReadout(varargin)

if nargin > 1
    DataPath = varargin{1};
    DataFile = varargin{2};
    % check if file actually exists
    FILES = getAllFiles(DataPath,'',0);
    if ~any(strcmpi(FILES,DataFile))
        error('Specified file does not exist in DataPath (maybe file ending missing)')
    end
    if nargin == 3
        opts = varargin{3};
    end
    if ~exist('opts') || ~isfield(opts,'device')
        choice = menu('Acquisition Device','Fukuda RADA','Verasonics','Ultrasonix','k-wave');
        switch choice
            case 1, opts.device = 'Fukuda';
            case 2, opts.device = 'Verasonics';
            case 3, opts.device = 'Ultrasonix';
            case 4, opts.device = 'kwave';
            case 0, return
        end
    end
elseif nargin > 0    
%%
%     [DataFile,DataPath,indx] = uigetfile(...
%             {[varargin{1} '\*.bin'],'Fukuda RADA (*.bin)'; ...
%             [varargin{1} '\*.dat'],'Verasonics (*.dat)';
%                [varargin{1} '\*.rf'],'Ultrasonix (*.rf)'}, ...
%                'Select a File',varargin{1});    
           [DataFile,DataPath,indx] = uigetfile(...
            {[varargin{1} '\*.*']}, ...
               'Select a File',varargin{1});
    if indx == 1,     opts.device = 'Fukuda';
    elseif indx == 2, opts.device = 'Verasonics';
    elseif indx == 3, opts.device = 'Ultrasonnix';   
    end
else 
    [DataFile,DataPath,indx] = uigetfile( ...
            {'*.bin','Fukuda RADA (*.bin)'; ...
            '*.dat','Verasonics (*.dat)';
               '*.rf','Ultrasonix (*.rf)'}, ...
               'Select a File');
    if indx == 1,     opts.device = 'Fukuda';
    elseif indx == 2, opts.device = 'Verasonics';
    elseif indx == 3, opts.device = 'Ultrasonnix';   
    end
end           

if ~any(strcmpi(DataPath(end),'\')) | ~any(strcmpi(DataPath(end),'/'))
    if ispc
        DataPath = [DataPath '\'];
    elseif isunix        
        DataPath = [DataPath '/'];
    end
end
if any(strcmpi(opts.device,'fukuda')) | any(strcmpi(opts.device,'Fukuda')) | any(strcmpi(opts.device,'UF760')) 
    [RF Bmode, opts] = FukudaReadout(DataPath,DataFile,opts);   
elseif any(strcmpi(opts.device,'verasonics')) | any(strcmpi(opts.device,'Verasonics'))
    [RF opts] = VerasonicsReadout(DataPath,DataFile,opts);
elseif any(strcmpi(opts.device,'Ultrasonix')) | any(strcmpi(opts.device,'ultrasonix')) 
    [RF opts] = UltrasonixReadout(DataPath,DataFile,opts);   
elseif any(strcmpi(opts.device,'kwave')) | any(strcmpi(opts.device,'kwave'))
    [RF opts] = kwaveReadout(DataPath,DataFile,opts);   
end

