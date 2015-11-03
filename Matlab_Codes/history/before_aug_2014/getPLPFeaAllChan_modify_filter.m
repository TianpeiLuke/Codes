%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The PLP feature extraction for all the data
%   include the filtering step
%  
%
%  written by Tianpei Xie, Mar_18_2013
%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

curpath = pwd; 
[dest_org, foldername, ext] = fileparts(curpath);
upper_org = '../../../Raw_data/Segmented_data';
upper_org2 = '../../Raw_data/Features_data'; %'../../../Raw_data/Features_data';

src_org =  strcat(upper_org, '/Filtered_data');
%dest_org = strcat(upper_org2, '/PLP_win186ms');
dest_org = strcat(upper_org2, '/PLP_envelop');
dest_dir = [{'/human_09/'}, ...
       {'/human_animal_09/'}, ...
       {'/human_10/'},...
       {'/human_animal_10/'} ];
   
src_dir = dest_dir;   
addpath('./rastamat');  
      
testChan  = [1:8, 10]; %[1 2 3 4 ];
fs = 10000;


%==============================================================
%To reduce the effect of manully work, we cannel the filtering step

%% construct the filterband 
% for human-animal, sensor 1-4,10
% a noise with periodic freq at  [1766,2647,3489,4348] with period
% be about 860 Hz
% bandstop with stopband at (1766 +- delta)
     shift = 2000;
     fupper = 1750+shift;
     flower = 1350+shift;
     [z,p,k] = butter(9,[flower, fupper]/fs, 'stop' );
     [sos,g] = zp2sos(z,p,k);
     Hd = dfilt.df2sos(sos,g);
     filterband(1) = {Hd};

%  bandstop with stopband at (2647 +- delta)
    shift2 = 2600;
    fupper = 2900+shift2;
    flower = 2500+shift2;
    [z2,p2,k2] = butter(9,[flower, fupper]/fs, 'stop' );
    [sos,g] = zp2sos(z2,p2,k2);
    Hd = dfilt.df2sos(sos,g);
    filterband(2) = {Hd};
     
    %lowpass to eliminate the constant noise at (3489 ) and (4348)
    shift = 3000;
    fcut = 3100+shift;
    [z,p,k] = butter(9,fcut/fs);
    [sos,g] = zp2sos(z,p,k);
    Hd = dfilt.df2sos(sos,g);
    filterband(3) = {Hd};
    
    % for seismic sensor 5-7 and PIR sensor 8
    shift = 50;
    fcut = 500+shift;
    [z,p,k] = butter(9,fcut/fs);
    [sos,g] = zp2sos(z,p,k);
    Hd = dfilt.df2sos(sos,g);
    filterband(4) = {Hd};
%================================================================
%% Parameter for MFCC    
  wintime =  0.186;%0.032;
 hoptime =  wintime*0.25; %0.025;%0.016; 
 numcep  = 13;
 nbands  = 40;
 maxfreq = 3000;  
 minfreq = 133.33;

%%
for folderid =1:length(dest_dir)
% loop over folders
    pathOrig = strcat(src_org, src_dir{folderid});
    pathDest = strcat(dest_org, dest_dir{folderid});
    listAllFiles = dir(pathOrig);
    nFiles = length(listAllFiles);
    display(['=====================================================']);
    display(['In folder ' pathDest]);
    display(['In folder ' pathOrig]);
    display(['Writing the log file...']);
% Create/Write the log file
    fid = fopen(strcat(pathDest ,'/log.txt'), 'w+');
    fprintf(fid, ' \n');
    fprintf(fid, '==================================================\n');
    time = clock;
    fprintf(fid, 'Date:\t %s\t Time:\t %d:%d:%2d \n', date, time(4),time(5),ceil(time(6)));
    fprintf(fid, 'CurrentPath:\t %s\n', curpath);
    fprintf(fid, 'Src_Filepath:\t  %s\n', pathOrig);
    fprintf(fid, 'Dest_Filepath:\t  %s\n', pathDest);
    fprintf(fid, 'Activity:\t  feature extraction: MFCC with filtering  \n');
    fprintf(fid, '-------------------------------------------------------------\n');
    fprintf(fid, 'Parameter:\n');
    fprintf(fid, '\t Window time(MFCC) (.sec): \t %.3f \t Hoptime: \t %.3f \n', wintime, hoptime);
    fprintf(fid, '\t Num of cepstal returned: \t %d \n', numcep);
    fprintf(fid, '\t Num of filter band: \t %d \n', nbands);
    fprintf(fid, '\t Maximal freq of filter band (Hz): \t %d \n', maxfreq);
    fprintf(fid, '-------------------------------------------------------------\n');
    fprintf(fid, 'Src_fullname \t  Dst_fullname \t Dimension of feature \t norm of feature \n ');
    fprintf(fid, '-------------------------------------------------------------\n');

  for i = 3:nFiles
    fullname = listAllFiles(i).name;        
    
    if strfind(fullname,'mat') ~= 0
        filename = fullname(1:end-4);
        indx = strfind(filename,'_');
        % Get the date
        preName = filename(1:indx(2)-1);
        % Get channel number
        chanNum = str2num(filename(indx(2)+5:indx(3)-1));
        % Get Segment number
        indx2 = strfind(filename,'Seg');
        segNum = filename(indx2+3:end);
        
        if ismember(chanNum, testChan)%strcmp(midName, 'Chan7')
            % Get path name
            fullpath = strcat(pathOrig, fullname)
            
            
            load(fullpath)
            
            
            %============== We cannel this step =========================
            % filter the data
            display('filter the data...');
            
            % at this step only for the same as human-animal
            data = aux_data_filter(data, chanNum, filterband);
            %============================================================ 
            % spectrual substration
            %data=SSBerouti79(data,fs);
            
            %===========================================================
            % extract the MFCC feature
            %[mm,aspc] = melfcc(data, fs, 'maxfreq', maxfreq, 'minfreq', minfreq,...
            % 'numcep', numcep, 'nbands', nbands, 'fbtype', 'fcmel', ...
            % 'dcttype', 1, 'usecmp', 1,...
            % 'wintime', wintime, 'hoptime', hoptime);
             [cep2, spec2] = rastaplp(data, fs, 0, 12);
             %[cep2, spec2] = rastaplp_mod(data, fs, 0, 12);
             
              del = deltas(cep2); 
            ddel = deltas(deltas(cep2,5),5);
            mm = [cep2;del;ddel];
            fea = reshape(mm, numel(mm),1);
            norm_fea = norm(fea);
%             iflage = 0;
%             if chanNum  == 1 && folderid == 2
%                 close all;
%                 figure(1)
%                 [S,F,T,P] = spectrogram(data,256,250,256,fs);
%                 surf(T,F,10*log10(P),'edgecolor','none'); axis tight; 
%                 view(0,90);
%                 xlabel('Time (Seconds)'); ylabel('Hz');
%                 
%                 output=SSBerouti79(data,10000);
%                 figure(2)
%                 [S,F,T,P] = spectrogram(output,256,250,256,fs);
%                 surf(T,F,10*log10(P),'edgecolor','none'); axis tight; 
%                 view(0,90);
%                 xlabel('Time (Seconds)'); ylabel('Hz');
%                 
%                 [im,ispc] = invmelfcc(mm, fs, 'maxfreq', maxfreq, 'minfreq', minfreq,...
%              'numcep', numcep, 'nbands', nbands, 'fbtype', 'fcmel', ...
%              'dcttype', 1, 'usecmp', 1,...
%              'wintime', wintime, 'hoptime', hoptime);
%                 figure(3)
%                 [S,F,T,P] = spectrogram(im,256,250,256,fs);
%                 surf(T,F,10*log10(P),'edgecolor','none'); axis tight; 
%                 view(0,90);
%                 xlabel('Time (Seconds)'); ylabel('Hz');
%                 if iflage==0
%                 
%                 %iflage = 1;
%                 pause;
%                 end
%             end
           
           feaName = strcat(pathDest,filename(1:indx2-1),'PLP_','Fea',segNum, '.mat');
           save(feaName, 'fea', 'norm_fea');        
            
           fprintf(fid, '%s \t %s \t %d \t %.3f \n', fullname, fullname, numel(mm), norm_fea);
           %
           display('end of feature extraction ');
           
           clear data  
           fprintf(fid, '---------------------------------------------------------\n');
        end 
    end
  end
  fprintf(fid, '---------------------------------------------------------\n');
  fprintf(fid, 'end of writting\n');
  fprintf(fid, ' \n');
  fclose(fid);
end