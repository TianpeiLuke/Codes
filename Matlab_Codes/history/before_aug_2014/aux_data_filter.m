function data_filter = aux_data_filter(data, chanNum, filterband)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Filter the data before feature extraction, the filter strategy depend on
%    the channel of the data
%  
% Input: 
%        data:        the original data from channel chanNum
%        chanNum:     channel number
%        filterband:  struct array containing all filter will be applied
%                     here, in particular, filterband(1:3) for human-animal
%                     for channel 1-4,10
%                     filterband(4) for human for channel 1-4,10, 
%                     filterband(5) for channel 5-8
%     
%
% Output: data_filter: the filtered data
%
%  Written by Tianpei Xie, Oct_30th_2012
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(data)
    error('Invalid data input');
end

if chanNum<0 || chanNum > 12
    errror('channel number should be within 1 to 12 ');
end


data_temp = data;

if chanNum <=4 || chanNum == 10
    % at freq band with periodic freq at  [1766,2647,3489,4348] with period
    % be about 860 Hz
    %bandstop with stopband at (1766 +- delta)
    %bandstop with stopband at (2647 +- delta)
    %lowpass to eliminate the constant noise at (3489 ) and (4348)

     % a lowpass filter with cutoff band 1500;
     % a cascade of filter bands 
     Hd = filterband{1};
     data_temp = filter(Hd,data_temp);
     Hd = filterband{2};
     data_temp = filter(Hd,data_temp);
     Hd = filterband{3};
elseif chanNum>= 5 && chanNum<=7
     Hd = filterband{4};
end

if ~ismember(chanNum, [8,9,11,12]) 
    data_filter = filter(Hd,data_temp); 
else
    data_filter = data;%do not filter the chan 8,9,11,12
end

