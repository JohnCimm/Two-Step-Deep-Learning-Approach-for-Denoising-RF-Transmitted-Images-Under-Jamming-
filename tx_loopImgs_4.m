clear all;

%% This is the transmitter code

rng(15);
% Parameters
Fs = 25e6;              % Sampling frequency (Hz)
symbolRate = 5e6;      % Symbol rate (QPSK symbols per second)
sps = Fs / symbolRate; % Samples per symbol. In this case, it is 5.
numSymbols = 12288;     % Number of symbols to transmit
%SNR = 20;              % Signal-to-noise ratio in dB
masterclockrate=25e6; %same as sampling rate% Create QPSK symbols (random symbol sequence)

% Transmit signal using the USRP (B200)
txRadio = comm.SDRuTransmitter( ...
    'Platform', 'B200', 'SerialNum', '33EE264', ...
    'CenterFrequency', 1e9, ...     % Center frequency (e.g., 1 GHz)
    'MasterClockRate', masterclockrate, ...
    'Gain', 40, ...                  % Transmitter gain
    'InterpolationFactor', 1);      % Interpolation factor to match sample rate

% Root Raised Cosine Filter parameters
rolloff = 0.25;  % Roll-off factor
filterSpan = 6;  % Filter span in symbols

% data = randi([0 3], numSymbols, 1); % Generate random symbols: 0, 1, 2, 3 (QPSK)
% load  .mat
cifarImage = load('data_batch_2.mat');
data = cifarImage.data;


% Set a loopTime to force each iteration of for loop to take a set amount
% of time, this ensures synchronization between radios.
xiaowenFirstLoopTime = 15;xiaowenLoopTime = 10.5;
connorFirstLoopTime = 20;connorLoopTime = 12;

if input('Jammer ran by Xiaowen(1) or Connor(0)?\n') == 1
    loopTime = xiaowenLoopTime;
    firstLoopTime = xiaowenFirstLoopTime;
else
    loopTime = connorLoopTime;
    firstLoopTime = connorFirstLoopTime;
end

startIdx = input('Starting image index: \n');
endIdx = input('What image do you want to end at?\n');

%load the pre-defined interleaver
load('Interleaver_dict_1000.mat');

%Allocate memory to save the original RF sequence
original_RF=zeros(endIdx - startIdx + 1,numSymbols*sps+sps*filterSpan); %length is 61470

pilot = randi([0,3],200,1);

for imageIndex = startIdx:endIdx
    tic;

    rawImg = data(imageIndex, :); % load original image
    img = reshape(rawImg, [32, 32, 3]);
    %change to Matlab format
    img = permute(img, [2, 1, 3]);
    img = uint8(img);


    imgVector = img(:);
    % Convert image pixels into bits (8 bits per pixel for simplicity)
    imgBits = de2bi(imgVector, 8, 'left-msb');  % Convert to binary (8 bits per pixel)
    % Flatten the binary matrix into a single vector
    imgBits = imgBits(:);

    numSymbols = length(imgBits) / 2;
    if mod(length(imgBits), 2) ~= 0
        error('Bit sequence length must be even for QPSK mapping.');
    end
    bitPairs = reshape(imgBits, 2, numSymbols)';

    qpskSymbols = bi2de(bitPairs, 'left-msb');

    perm_indices = Interleave_dict(imageIndex -startIdx +1, :); %pick  a random  interleaver
    Interleave_qpskSymbols = qpskSymbols(perm_indices); %shuffle the qpsk sequence
    pilotSymbols = [pilot;Interleave_qpskSymbols];

    % QPSK Modulation
    qpskMod = comm.QPSKModulator('PhaseOffset', pi/4); % QPSK with a 45-degree phase offset
    %modulatedSignal = qpskMod(qpskSymbols); % Modulate data
    modulatedSignal = qpskMod(pilotSymbols); % Modulate the interleaved QPSK symbols
    

    up_modulatedSignal= upsample(modulatedSignal, sps); %upsampling,  insert zeros
    h = rcosdesign(rolloff, filterSpan, sps);  %design root raised cosine filter

    uy = upfirdn(up_modulatedSignal, h); %Apply RRC at the transmiter side

    % Create a chirp signal to begin the transmission, this will act as a
    % signal that the sequence has started.
    chirp_len=500;

    % generate chirp signal
    f0 = 0.5e6;
    f1 = 2e6;
    T = chirp_len/Fs; %time duation of chirp signal

    % Time vector
    t = [0:1:chirp_len-1]/Fs;

    % Chirp signal, column vector
    %scale by sqrt(2) so that the average power of the chirp signal is 1.
    chirp_signal = sin(2*pi* (f0 + (f1 - f0) .* t / T) .* t)';

    uy=[chirp_signal;uy];                           % total uy - length 62370

    % Create a repetition of the signal to transmit to make it easier to
    % receive the desired signal. the 5000 repetition gives about a 12 second
    % transmission (12288*5+500)*5000/25/10^6=12s
    % pause(12);                               % Connor use 12

    repeatedSig = repmat(uy,2000,1);

    % First transmission always takes the longest, if this is done now future
    % transmissions will execute in more consistent timing.


    if imageIndex > startIdx
        pause(3.5);
    end

    txRadio(repeatedSig);

    disp(['Transmitted image: ',num2str(imageIndex)]);
    
    if imageIndex > startIdx
        rem = loopTime - toc;
    else
        rem = firstLoopTime - toc;
    end
    pause(rem);
end

release(txRadio);