[y, Fs] = audioread('twostompclap.wav');
freq = 3.25 - 1.3;
x = y(1:156000, 2).';

units = 20;
count = 8;
index = 1;
impulse = zeros(1, freq*Fs*(count-1));
for i = 1:count
    for j = 0:units
        impulse((i-1)*(freq)*Fs+j*1 + 1) = 1;
    end
end

out = conv(impulse, x);
audiowrite('output.wav',out,Fs);

%The orignal sound clip is read and then truncated to 3.25 seconds. I found
%that to get the stompostompclap portion on beat, I needed to take another
%1.3 seconds off of the frequency the clip would play. I then constructed a
%impulse train with period 3.25-1.3 and repeated it 10 times (and offset by
%different amounts) to get the echo after convolving with the sound clip%