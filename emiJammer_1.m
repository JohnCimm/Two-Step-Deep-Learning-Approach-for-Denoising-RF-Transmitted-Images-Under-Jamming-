clear all;
close all;
% MATLAB code to generate and transmit a sinusoidal signal using USRP B200

% Parameters
Fs = 25e6;         % Sampling frequency (50 MHz)
fcut = 9e6;          % Low pass filter Cutoff frequency in Hz
sampleRate = Fs;  % Sample rate for USRP
centerf=1.0e9;
masterclockrate=25e6;

load('EMISig1.mat');
load('lengthArrayEMISignal1.mat');


% create loop to use random frames
sigma = input('Noise level:\n'); %set initial sigma to be 0.32
dur = input('How many images are you collecting? \n');
durSec = dur*30;
totalTime = tic;
loopPeriod = 30;

rng(15); % define random seed

while toc(totalTime) < durSec
    
    loopTime = tic;
    %extract signal from a certain frame
    frame_index=randi([500 580]);

    start_index=sum(lengths(1:frame_index-1))+1;
    end_index=start_index + lengths(frame_index)-1;

    jammingSignal=EMISig1(start_index: end_index);

    symbolRate=Fs;

    %scaling
    % jammingSignal = jammingSignal / max(abs(jammingSignal));

    %plot fft of original signal
%     L = length(jammingSignal);
%     freqs = symbolRate*(-L/2:L/2-1)/L;
%     time = (0:L-1)*(1/symbolRate);
%     fftdata = fftshift(fft(jammingSignal));
%     ft = abs(fftdata);
%     figure(1);
%     plot(freqs, ft);
%     title('Freq Domain original complex signal');


    %low pass filtering with a cut off frequency at fcut=9MHz, apply
    %scaling
    filtered_signal= lowpass(jammingSignal,fcut,Fs)*sigma;
    filtered_signal = filtered_signal.';
    

    %plot fft of filtered signal
%     L = length(filtered_signal);
%     freqs = symbolRate*(-L/2:L/2-1)/L;
%     time = (0:L-1)*(1/symbolRate);
%     fftdata = fftshift(fft(filtered_signal));
%     ft = abs(fftdata);
%     figure(2);
%     plot(freqs, ft);
%     title('Freq Domain filtered complex signal');



    % Configure the USRP B200
    txRadio = comm.SDRuTransmitter( ...
        'Platform', 'B200', ...
        'SerialNum', '326F067', ... % IP address of USRP
        'CenterFrequency', centerf, ...    % Center frequency (1 GHz)
        'MasterClockRate', masterclockrate, ...
        'Gain', 40, ...                   % Transmitter gain
        'InterpolationFactor', 1);        % No interpolation for real-time signal

    %need to modify to send jamming signals from different frames
    
    % Need to do testing to adjust timing accordingly - want to loop each
    % time in 14 seconds. This will adjust the amount of loops completed.

    for i = 1:1000
        % Transmit the jamming signal
        % disp('Transmitting jamming signal...');
        txRadio(filtered_signal);
    end

    % Release the transmitter after transmission
    release(txRadio);
    rem = loopPeriod - toc(loopTime);
    pause(rem);
end