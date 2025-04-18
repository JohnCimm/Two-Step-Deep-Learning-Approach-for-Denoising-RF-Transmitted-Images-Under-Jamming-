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
% Configure the USRP B200
txRadio = comm.SDRuTransmitter( ...
    'Platform', 'B200', ...
    'SerialNum', '326F067', ... % IP address of USRP
    'CenterFrequency', centerf, ...    % Center frequency (1 GHz)
    'MasterClockRate', masterclockrate, ...
    'Gain', 40, ...                   % Transmitter gain
    'InterpolationFactor', 1);        % No interpolation for real-time signal

% create loop to use random frames
sigma = input('Noise level:\n');
dur = input('How many images are you collecting? \n');
totalTime = tic;
firstLoopPeriod = 30;
loopPeriod = 30;
durSec = dur*loopPeriod+firstLoopPeriod - loopPeriod;

jammerFrameRan = zeros(dur,1);

rng(5);

count = 1;
while toc(totalTime) < durSec

    loopTime = tic;
    %extract signal from a certain frame
    frame_index=randi([500 580]);
    jammerFrameRan(count,1) = frame_index;

    start_index=sum(lengths(1:frame_index-1))+1;
    end_index=start_index + lengths(frame_index)-1;

    jammingSignal=EMISig1(start_index: end_index);

    filtered_signal= lowpass(jammingSignal,fcut,Fs)*sigma;
    filtered_signal = filtered_signal.';

    % Need to do testing to adjust timing accordingly - want to loop each
    % time in 30 seconds. This will adjust the amount of loops completed.
    if toc(totalTime) < firstLoopPeriod
        pause(2.5);
    end

    for i = 1:500
        % Transmit the jamming signal
        % disp('Transmitting jamming signal...');
        txRadio(filtered_signal);
    end

    % Release the transmitter after transmission
    release(txRadio);

    if toc(totalTime) < firstLoopPeriod
        rem = firstLoopPeriod - toc(loopTime);
    else
        rem = loopPeriod - toc(loopTime);
    end
    count = count + 1;
    pause(rem);
end

in = input('Want to save frames\n','s');

if in == 'y'
    outputFolder = 'rng10Frames';
    if ~exist(outputFolder,'dir')
        mkdir(outputFolder);
    end

    save(fullfile(outputFolder,'frames_Jammer.mat'),'jammerFrameRan','-v7.3');
    
end