clear vars;

%% This is the receiver code

rng(15);

% Parameters
Fs = 25e6;              % Sampling frequency (Hz)
symbolRate = 5e6;      % Symbol rate (QPSK symbols per second)
sps = Fs / symbolRate; % Samples per symbol. In this case, it is 5.
numSymbols = 12288;     % Number of symbols to transmit, corresponds to image 32*32*3*8/2
%SNR = 20;              % Signal-to-noise ratio in dB
masterclockrate=25e6; %same as sampling rate
numChunks = 5; %if numSymbols is large, then we need a larger numChunks  in order to fit the data within a chunk
chunkSize = 1e5;
filterSpan = 6;


% Receive the signal using the USRP (B200)
rxRadio = comm.SDRuReceiver( ...
    'Platform', 'B200', 'SerialNum', '326F0BC', ...
    'CenterFrequency', 1e9, ...     % Center frequency (e.g., 1 GHz)
    'Gain', 40, ...                  % Receiver gain
    'MasterClockRate', masterclockrate, ...
    'SamplesPerFrame', chunkSize, ...      % Number of samples per frame
    'DecimationFactor', 1, ...      % Decimation factor to match sample rate
    'OutputDataType', 'double');     % Output data type


%prepare orignial image
cifarImage = load('data_batch_2.mat');
imgData = cifarImage.data;

% Set a loopTime to force each iteration of for loop to take a set amount
% of time, this ensures synchronization between radios.
xiaowenFirstLoopTime = 14.5;xiaowenLoopTime = 10.5;
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

noisy_RF = zeros(endIdx-startIdx,61470);
noisy_RF_manualCFO = zeros(endIdx-startIdx,61470);
CFO_data = zeros(endIdx-startIdx,2);
original_RF = zeros(endIdx - startIdx + 1,61470);
correlations_RF = zeros(endIdx-startIdx,chunkSize*2-1);
correlations_RF_manual = zeros(endIdx-startIdx,chunkSize*2-1);
angles = zeros(endIdx-startIdx,4);

N = endIdx-startIdx+1;
H = 32;
W = 32;
C = 3;
noisy_imgs = zeros(N,H,W,C);
orig_imgs = zeros(N,H,W,C);

%load the pre-defined interleaver
load('Interleaver_dict_1000.mat');

pilot = randi([0,3],200,1);

for imageIndex = startIdx:endIdx
    tic;

    rawImg = imgData(imageIndex, :); % load original image
    img0 = reshape(rawImg, [32, 32, 3]);
    orig_imgs(imageIndex-startIdx+1,:,:,:) = img0;
    %change to Matlab format
    img1 = permute(img0, [2, 1, 3]);
    img = uint8(img1);

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
    %end of re-generating the original image transmission data

    %data comes from the image
    data = qpskSymbols; %the number of qpskSymbols should be 12288.

    perm_indices = Interleave_dict(imageIndex -startIdx +1, :); %pick  a random  interleaver
    Interleave_qpskSymbols = data(perm_indices); %shuffle the qpsk sequence

    pilotSymbols = [pilot;Interleave_qpskSymbols];

    % QPSK Modulation
    qpskMod = comm.QPSKModulator('PhaseOffset', pi/4); % QPSK with a 45-degree phase offset

    %modulatedSignal = qpskMod(data); % Modulated data complex valued
    modulatedSignal = qpskMod(pilotSymbols); % Modulate the interleaved QPSK symbols

    receivedSignal_all=zeros(chunkSize*numChunks,1);
    receivedSignal=zeros(chunkSize,1);

    % Create QPSK symbols (random bit sequence)
    % data = randi([0 3], numSymbols, 1); % Generate random symbols: 0, 1, 2, 3 (QPSK)


    % rng(55);
    % data = randi([0 3], numSymbols, 1); % Generate random symbols: 0, 1, 2, 3 (QPSK)
    %data = data';

    % Raised Root Cosine Filter Design
    rolloff = 0.25;  % Roll-off factor
    filterSpan = 6;  % Filter span in symbols

    % Apply the Raised Root Cosine Filter
    %filteredSignal = rrcFilter(modulatedSignal);

    up_modulatedSignal= upsample(modulatedSignal, sps); %upsampling,  insert zeros
    h = rcosdesign(rolloff, filterSpan, sps);  %design root raised cosine filter

    uy = upfirdn(up_modulatedSignal, h); %Apply RRC at the transmiter side
    original_RF(imageIndex -startIdx +1 ,:)=uy(sps*length(pilot)+1:length(uy)).';

    extractedPilot = uy(1:sps*length(pilot));


    chirp_len=500;  %length of chirp signal
    %chirp_signal = ones(chirp_len, 1);
    %uy = [ones(100,1);uy];

    %generate chirp signal
    f0 = 0.5e6;
    f1 = 2e6;
    T = chirp_len/Fs; %chirp signal time duration

    % Time vector
    t = [0:1:chirp_len-1]/Fs;

    % Chirp signal, column vector
    %scale by sqrt(2) so that the average power of the chirp signal is 1.
    chirp_signal = sin(2*pi* (f0 + (f1 - f0) .* t / T) .* t)';

    uy = [chirp_signal;uy];                         % total uy - length 62370

    % Prompt the user to start receiving, user should input 'y' and press enter
    % at the same time as the transmitter prompt to ensure the receiver is
    % receiving at the same time as the transmission is happening.

    if imageIndex == startIdx
        pause(4.5);
    end

    for i = 1:numChunks
        chunk = rxRadio();
        startIndex = (i - 1) * chunkSize + 1;
        endIndex = i * chunkSize;
        receivedSignal_all(startIndex:endIndex) = chunk;
    end


    release(rxRadio);

    % close all;
    %process data chunks one by one to see whether data can be found in one of
    %the chunks
    for i=1:numChunks

        if i == 3
            continue
        end
        startIndex = (i - 1) * chunkSize + 1;
        endIndex = i * chunkSize;
        receivedSignal=receivedSignal_all(startIndex:endIndex);

        %%%% first step is to do CFO compensation
        freqComp = comm.CoarseFrequencyCompensator(...
            'Modulation', 'QPSK',...
            'SampleRate', Fs);
        [receivedSigComp1 estimateOffset] = freqComp(receivedSignal);
        
        %if (-estimateOffset < 350) && (-estimateOffset > 210)
           
        %else
            cfo = -262.2604;                % -286.1023, -262.2604, -309.9442 -333.786? (23.8418 drift)
            t = linspace(0,length(receivedSignal)/Fs,length(receivedSignal));
            receivedSignal_manual = receivedSignal.*exp(-j*2*pi*cfo.*t');
%% All this code just to confirm manual CFO works            
            receivedSignalCorr = receivedSignal_manual;

        %time synchronization using chirp signal
        correlation = xcorr(receivedSignalCorr, chirp_signal);          % Find time location
        correlations_RF_manual(imageIndex - startIdx+1,:) = correlation.';

        % Extract the index of the peak
        upplimit= 2*chunkSize - numSymbols * sps - 2*chirp_len - 2*sps*length(pilot)-1;
        lowlimit= chunkSize;
        %[peakvalue, peakIndex] = max(abs(correlation(1:130000)));
        %[peakvalue, peakIndex] = max(abs(correlation(1:uplimit)));
        [peakvalue, peakIndex] = max(abs(correlation(lowlimit:upplimit)));
        peakIndex = peakIndex + lowlimit -1;

        %Need to lower the threshold for the correlation threshold when there is jamming
        % threshold = chirp_len *0.08*2;
        threshold = median(abs(correlation(chunkSize:2*chunkSize-1)));  % Use this when jamming

        if (peakvalue > 10*threshold) %process this chunk only if the correlation is large enough
            % correlation function outputs larger array, shift to actual position.
            startIndex = peakIndex - length(receivedSignal_manual) + 1;
            endIndex = startIndex+length(uy)-1;
            %disp(startIndex);

            % Extract one clean instance of 'uy'
            capturedSignal_Manual = receivedSignal_manual(startIndex:endIndex);
            capturedSignal2 = receivedSignal_manual(endIndex+1:endIndex+1+chirp_len+sps*length(pilot));

            % estimate the initial phase shift using pilot signal
            pilotStart = chirp_len+1;
            pilotEnd = pilotStart+sps*length(pilot)-1;

            angle1_manual = angle(extractedPilot'*capturedSignal_Manual(pilotStart:pilotEnd));
            angle2_manual = angle(extractedPilot'*capturedSignal2(pilotStart:pilotEnd));

            
            angles(imageIndex-startIdx+1,1) = angle(chirp_signal'*capturedSignal_Manual(1:chirp_len));
            angles(imageIndex-startIdx+1,2) = angle1_manual;
            angles(imageIndex-startIdx+1,3) = angle(chirp_signal'*capturedSignal2(1:chirp_len));
            angles(imageIndex-startIdx+1,4) = angle2_manual;
            
            phaseOffset_manual = angle((exp(j*angle1_manual) + exp(j*angle2_manual))/2);

            disp(['Phase Offset Manual CFO: ', num2str(phaseOffset_manual)]);



            % Now compensate the received signal with the appropriate phase offset

            recSigComp_manual = capturedSignal_Manual*exp(-j*phaseOffset_manual);

            % Process the actual data, ignore the chirp (first chirp_len signals)
            %dataRec = receivedSigComp(101:length(receivedSigComp));
            dataRec_manual = recSigComp_manual(chirp_len+length(pilot)*sps+1:length(recSigComp_manual));

            %save the noisy RF signal
            noisy_RF_manualCFO(imageIndex-startIdx+1,:) = dataRec_manual.';


            uy2 = upfirdn(dataRec_manual, h); %Apply RRC at the receiver side after CFO compensation
            dy = downsample(uy2,sps); %downsample uy2 to recover orginal x
            

            % QPSK Demodulation
            qpskDemod = comm.QPSKDemodulator('PhaseOffset', pi/4); % Same phase offset as the modulator
            demodulatedData = qpskDemod(dy);

            demodulatedData_1= demodulatedData(1+filterSpan : numSymbols+filterSpan);
            de_demodulatedData_1(perm_indices)=demodulatedData_1;


            %convert demodulated data back to an received image
            %receivedBits = de2bi(demodulatedData(1+filterSpan : numSymbols+filterSpan), 2, 'left-msb');
            receivedBits = de2bi(de_demodulatedData_1, 2, 'left-msb');
            receivedBits = receivedBits';
            receivedBits = receivedBits(:);

            % Convert Bits Back to Image Vector
            receivedBits = reshape(receivedBits, [], 8); % Reshape into bytes
            receivedImageVector = bi2de(receivedBits, 'left-msb');
            receivedImageVector = uint8(receivedImageVector); % Convert to 8-bit unsigned integers

            % Reshape to Image
            receivedImage_manual = reshape(receivedImageVector, [32, 32, 3]);            

        end
%% Resume normal code

           receivedSignal = receivedSigComp1;

           CFO_data(imageIndex - startIdx+1,1) = imageIndex;
           CFO_data(imageIndex - startIdx+1,2) = estimateOffset;
        %end

        % Get average signal strength
        % energy = receivedSignal'*receivedSignal;
        % avgPower = energy/length(receivedSignal);
        % disp(avgPower);
        % Normalize signal

        receivedSignalCorr = receivedSignal;

        %time synchronization using chirp signal
        correlation = xcorr(receivedSignalCorr, chirp_signal);
        correlations_RF(imageIndex - startIdx+1,:) = correlation.';

        % Extract the index of the peak
        upplimit= 2*chunkSize - numSymbols * sps - 2*chirp_len - 2*sps*length(pilot)-1;
        lowlimit= chunkSize;
        %[peakvalue, peakIndex] = max(abs(correlation(1:130000)));
        %[peakvalue, peakIndex] = max(abs(correlation(1:uplimit)));
        [peakvalue, peakIndex] = max(abs(correlation(lowlimit:upplimit)));
        peakIndex = peakIndex + lowlimit -1;

        %Need to lower the threshold for the correlation threshold when there is jamming
        % threshold = chirp_len *0.08*2;
        threshold = median(abs(correlation(chunkSize:2*chunkSize-1)));  % Use this when jamming

        if (peakvalue > 10*threshold) %process this chunk only if the correlation is large enough
            % correlation function outputs larger array, shift to actual position.
            startIndex = peakIndex - length(receivedSignal) + 1;
            endIndex = startIndex+length(uy)-1;
            %disp(startIndex);

            % Extract one clean instance of 'uy'
            capturedSignal = receivedSignal(startIndex:endIndex);
            capturedSignalNext = receivedSignal(endIndex+1:endIndex+1+chirp_len+sps*length(pilot));
            receivedSigComp = capturedSignal;

            % estimate the initial phase shift using pilot signal
            pilotStart = chirp_len+1;
            pilotEnd = pilotStart+sps*length(pilot)-1;

            angle1 = angle(extractedPilot'*capturedSignal(pilotStart:pilotEnd));
            angle2 = angle(extractedPilot'*capturedSignalNext(pilotStart:pilotEnd));

            phaseOffset = angle((exp(j*angle1) + exp(j*angle2))/2);

            disp(['Phase Offset Matlab CFO: ', num2str(phaseOffset)]);


            % Now compensate the received signal with the appropriate phase offset
            receivedSigComp = receivedSigComp*exp(-j*phaseOffset);

            % Process the actual data, ignore the chirp (first chirp_len signals)
            %dataRec = receivedSigComp(101:length(receivedSigComp));
            dataRec = receivedSigComp(chirp_len+sps*length(pilot)+1:length(receivedSigComp));
            
            close all; 
            
            figure;
            scatterplot(dy);

            %save the noisy RF signal
            noisy_RF(imageIndex -startIdx +1 ,:)=dataRec.'; %save the RF signal, length is 61470

            uy2 = upfirdn(dataRec, h); %Apply RRC at the receiver side after CFO compensation
            dy = downsample(uy2,sps); %downsample uy2 to recover orginal x


            % QPSK Demodulation
            qpskDemod = comm.QPSKDemodulator('PhaseOffset', pi/4); % Same phase offset as the modulator
            demodulatedData = qpskDemod(dy);

            % Compute and display the Symbol Error Rate (SER)

            %ser = sum(data ~= demodulatedData(1+filterSpan : numSymbols+filterSpan)) / numSymbols;
            ser = sum(Interleave_qpskSymbols ~= demodulatedData(1+filterSpan : numSymbols+filterSpan)) / numSymbols;

            % Scatter plot of the received signal prior to demodulation
            %figure('Scatter');
            scatfig = scatterplot(dy);

            est_amp = sqrt(dy(1+filterSpan : numSymbols+filterSpan)'*dy(1+filterSpan : numSymbols+filterSpan)/numSymbols);
            signal_diff= dy(1+filterSpan : numSymbols+filterSpan)/est_amp - modulatedSignal(length(pilot)+1:numSymbols+length(pilot));
            MSE = signal_diff'*signal_diff/numSymbols;

            % disp(['total # of symbols received: ', num2str(numSymbols)]);
            % disp(['Chirp sequence length: ', num2str(chirp_len)]);
            disp(['receive data at chunk index: ', num2str(i), '    Estimated CFO ', num2str(estimateOffset)]);
            disp(['Sync. startIndex ', num2str(startIndex),'            Sync. peakvalue ', num2str(peakvalue)]);
            disp(['MSE: ', num2str(MSE),'                     Symbol Error Rate: ', num2str(ser)]);

            %need to de-interleave the data before converting back to an image
            demodulatedData_1= demodulatedData(1+filterSpan : numSymbols+filterSpan);
            de_demodulatedData_1(perm_indices)=demodulatedData_1;


            %convert demodulated data back to an received image
            %receivedBits = de2bi(demodulatedData(1+filterSpan : numSymbols+filterSpan), 2, 'left-msb');
            receivedBits = de2bi(de_demodulatedData_1, 2, 'left-msb');
            receivedBits = receivedBits';
            receivedBits = receivedBits(:);

            % Convert Bits Back to Image Vector
            receivedBits = reshape(receivedBits, [], 8); % Reshape into bytes
            receivedImageVector = bi2de(receivedBits, 'left-msb');
            receivedImageVector = uint8(receivedImageVector); % Convert to 8-bit unsigned integers

            % Reshape to Image
            receivedImage = reshape(receivedImageVector, [32, 32, 3]);

            % Display transmited image and received image side by side
            figure(1);
            subplot(1,3,1);
            imshow(img);
            title('Transmitted Image');

            subplot(1,3,2);
            imshow(receivedImage);
            title('Received Image');

            subplot(1,3,3);
            imshow(receivedImage_manual);
            title('manual CFO');

            %calculate two metrics PSNR and SSIM to compare the transmitted image and the received image.
            % The higher the PSNR, the better. An SSIM value of 1 is the best.
            [peaksnr, snr] = psnr(receivedImage, img);
            [ssimval, ssimmap] = ssim(receivedImage,img);
            disp(['PSNR: ', num2str(peaksnr),'  SSIM value: ', num2str(ssimval)]);

            noisy_imgs(imageIndex-startIdx +1,:,:,:) = receivedImage;


            break;
        end %finish processing the chunk that contains the transmitted data

        if (i==numChunks)
            disp('Did not locate data, increase numChunks or decrease correlation threshold');
        end

    end %finish processing all chunks

    if imageIndex > startIdx
        rem = loopTime - toc;
    else
        rem = firstLoopTime - toc;
    end
    pause(rem);
end


%Save images in folder if desired
in = input('Want to save data?\n','s');

if in == 'y'
    outputFolder = input('output folder: \n','s');
    dir = 'C:\Users\Test\Desktop\L3Harris Project\RFData';
    fullFile = [dir,'\',outputFolder];
    if ~exist(fullFile,'dir')
        mkdir(fullFile);
    end

    save([fullFile,'\','orig_imgs.mat'],'orig_imgs','-v7.3');
    save([fullFile,'\','noisy_imgs.mat'],'noisy_imgs','-v7.3');
    save([fullFile,'\','noisy_RF.mat'], 'noisy_RF', '-v7.3');
    save([fullFile,'\','noisy_RF_manual.mat'],'noisy_RF_manualCFO','-v7.3');
    save([fullFile,'\','CFO_data.mat'],'CFO_data','-v7.3');
    save([fullFile,'\','original_RF.mat'], 'original_RF', '-v7.3');
    save([fullFile,'\','correlations_RF.mat'],'correlations_RF','-v7.3');
    save([fullFile,'\','correlations_RF_manual.mat'],'correlations_RF_manual','-v7.3');
    save([fullFile,'\','angles.mat'],'angles','-v7.3');


    if input('Want to send to Jetson?\n','s') == 'y'
        % Server address and port
        % url = 'http://192.168.1.192:5000/unet'; %house

        % url = 'http://10.18.206.229:5000/unet'; %u of u wifi

        url = 'http://10.56.52.164:5000/unet'; % uGuest

        %'http://192.168.1.192:5000/unet' %house

        % File paths for the .mat files to upload
        file1_path = fullfile(outputFolder,'\orig_imgs.mat'); % Path to the first .mat file
        file2_path = fullfile(outputFolder,'\noisy_imgs.mat');  % Path to the second .mat file

        % Construct the curl command for making a POST request with file upload
        command = sprintf(['curl -X POST %s ' ... % Specify POST method
            '-F "file1=@%s" ' ... % Attach the first file as "file1"
            '-F "file2=@%s"'], url, file1_path, file2_path); % Attach the second file as "file2"

        % Execute the curl command
        [status, cmdout] = system(command); % Run the system command and capture the status and output

        % Check the status of the command execution
        if status == 0
            disp('Response:'); % Display the server response if successful
            disp(cmdout); % Print the response content
        else
            disp('Error occurred:'); % Display an error message if the command fails
            disp(cmdout); % Print the error details
        end
    end

end
