% test script
%
% 
% 
clc
clear all;

[signal,fs,nbits]=wavread('20091209200205_set3_Chan2_full.wav');
output=SSBerouti79(signal,fs);
%hfile('20091209200205_set3_Chan2_full_c.wav');
wavwrite(output,fs,'20091209200205_set3_Chan2_full_c.wav')

[S,F,T,P] =  spectrogram(output,256,250,256,fs);
surf(T,F,10*log10(P),'edgecolor','none'); axis tight; 
view(0,90);
xlabel('Time (Seconds)'); ylabel('Hz');