function [iscorrupt, index]= corrupt_infer(preName, chanNum)
% judge if the data is corrupted
if chanNum ~=3 && chanNum ~=4
  iscorrupt = 0;
end

if chanNum == 3
filename1 = './chan3_descrp.txt';
fid = fopen(filename1);
textstring = textscan(fid, '%d %s %s %s', 'delimiter', '\t');
fclose(fid);

% ---------------read index ---------------
ind = find(strncmp(preName, textstring{2},14)>0);
if length(ind)== 2
   if str2num(preName(end)) == 3
       ind(2) = [];
   else
       ind(1) = [];
   end
end
index = textstring{1}(ind);
iscorrupt = strcmp('B',textstring{4}{ind});
%----------------

elseif chanNum == 4

filename1 = './chan4_descrp.txt';
fid = fopen(filename1);
textstring = textscan(fid, '%d %s %s %s', 'delimiter', '\t');
fclose(fid);
% ---------------read index ---------------
ind = find(strncmp(preName, textstring{2},14)>0);
if length(ind)== 2
   if str2num(preName(end)) == 3
       ind(2) = [];
   else
       ind(1) = [];
   end
end
index = textstring{1}(ind);
iscorrupt = strcmp('B',textstring{4}{ind});
else
 % ---------------read index ---------------
 filename1 = './chan4_descrp.txt';
fid = fopen(filename1);
textstring = textscan(fid, '%d %s %s %s', 'delimiter', '\t');
fclose(fid);
ind = find(strncmp(preName, textstring{2},14)>0);
if length(ind)== 2
   if str2num(preName(end)) == 3
       ind(2) = [];
   else
       ind(1) = [];
   end
end
index = textstring{1}(ind); 

end





