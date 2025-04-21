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


%prepare orignal image
cifarImage = load('data_batch_2.mat');
imgData = cifarImage.data;

% Set a loopTime to force each iteration of for loop to take a set amount
% of time, this ensures synchronization between radios.
xiaowenFirstLoopTime = 14.5;xiaowenLoopTime = 10.5;
connorFirstLoopTime = 30;connorLoopTime = 30;                               % Adjust these two numbers to change the loop time of Matlab code

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
original_RF = zeros(endIdx - startIdx + 1,61470);


N = 1;
H = 32;
W = 32;
C = 3;
orig_imgs = zeros(N,H,W,C);
noisy_imgs = zeros(N,H,W,C);

%load the pre-defined interleaver
load('Interleaver_dict_1000.mat');

pilot = randi([0,3],200,1);

sending = (input('Sending Images? \n','s') == 'y');

if sending
    %Save images in folder if desired
    outputFolder = input('output folder: \n','s');
    dir = 'C:\Users\Test\Desktop\L3Harris Project\RFData';
    fullFile = [dir,'\',outputFolder];
    if ~exist(fullFile,'dir')
        mkdir(fullFile);
    end
end


for imageIndex = startIdx:endIdx
    tic;

    rawImg = imgData(imageIndex, :); % load original image
    img0 = reshape(rawImg, [32, 32, 3]);
    %change to Matlab format
    img1 = permute(img0, [2, 1, 3]);
    img = uint8(img1);
    orig_imgs(1,:,:,:) = imrotate(img,90);

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

    perm_indices = Interleave_dict(imageIndex, :); %pick  a random  interleaver
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
    origRF=uy(sps*length(pilot)+1:length(uy)).';

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

        reg = [-262.2604, -286.1023, -333.786, -357.6278];
        estimateOffset = round(estimateOffset,4);

        if any(reg == estimateOffset)
            receivedSignal = receivedSigComp1;
        else
            cfo = -262.2604;                % -286.1023, -262.2604, -309.9442 -333.786? (23.8418 drift)
            t = linspace(0,length(receivedSignal)/Fs,length(receivedSignal));
            receivedSignal = receivedSignal.*exp(-j*2*pi*cfo.*t');

            %disp(['Using Manual CFO: ', num2str(cfo)]);
        end


        receivedSignalCorr = receivedSignal;

        %time synchronization using chirp signal
        correlation = xcorr(receivedSignalCorr, chirp_signal);

        % Extract the index of the peak
        upplimit= 2*chunkSize - numSymbols * sps - 2*chirp_len - 2*sps*length(pilot)-1;
        lowlimit= chunkSize;
        %[peakvalue, peakIndex] = max(abs(correlation(1:130000)));
        %[peakvalue, peakIndex] = max(abs(correlation(1:uplimit)));
        [peakvalue, peakIndex] = max(abs(correlation(lowlimit:upplimit)));
        [totalPk, totalPkIdx] = max(abs(correlation));
        peakIndex = peakIndex + lowlimit -1;

        %Need to lower the threshold for the correlation threshold when there is jamming
        % threshold = chirp_len *0.08*2;
        threshold = .90*totalPk;

        if (peakvalue > threshold) %process this chunk only if the correlation is large enough
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

            %disp(['Phase Offset: ', num2str(phaseOffset)]);


            % Now compensate the received signal with the appropriate phase offset
            receivedSigComp = receivedSigComp*exp(-j*phaseOffset);

            % Process the actual data, ignore the chirp (first chirp_len signals)
            %dataRec = receivedSigComp(101:length(receivedSigComp));
            dataRec = receivedSigComp(chirp_len+sps*length(pilot)+1:length(receivedSigComp));

            %save the noisy RF signal
            noisyRF = dataRec.';

            %% Send Sequence to Jetson

            %convert sequence from length 61470 to sequences of length 5120
            %energy scaling
            E_noiseSeq=abs(noisyRF*noisyRF')/61470;
            E_originalSeq=abs(origRF*origRF')/61470;
            %averge energy of the original signal is 0.1988
            uy_noisy_scaled = noisyRF*sqrt(0.1988/E_noiseSeq);

            for k=1:12 %12 blocks of 5120 to get 5120*12=61440, but uy has a length of 61470
                original_Seq(k,:)=origRF((k-1)*5120+1: k*5120).';
                noisy_Seq(k,:)=uy_noisy_scaled((k-1)*5120+1: k*5120).';
            end

            %this is to take care of the additional 61470-61440=30 samples
            original_Seq(13,:)=origRF(61470-5120+1: 61470).';
            noisy_Seq(13,:)=uy_noisy_scaled(61470-5120+1: 61470).';

            file1_path = [fullFile,'\','original_test_hardware_1.mat'];
            file2_path = [fullFile,'\','noisy_test_hardware_1.mat'];

            save(file1_path, 'original_Seq', '-v7.3');
            save(file2_path, 'noisy_Seq', '-v7.3');

            url = 'http://192.168.1.10:5000/unet';

            % File paths for the .mat files to upload

            % Path to save the returned .mat file
            output_path = 'C:\Users\Test\Documents\MATLAB\L3Harris Project\working_singleImg\received_result.mat';

            % Enable inference mode (set to 'true' or 'false')
            inference_flag = 'true';

            send(file1_path,file2_path,imageIndex);
            % The 'Noised, denoised, original' Should be outputs from the
            % Jetson device, the first dimension should be 1 since we are
            % sending one image per loop.
            load('received_result.mat');
            noisy_Seq(:,:) = squeeze(noised(:,1,:) + 1i * noised(:,2,:));            % From Jetson
            denoised_Seq(:,:) = squeeze(denoised(:,1,:) + 1i * denoised(:,2,:));     % From Jetson
            original_Seq(:,:) = squeeze(original(:,1,:) + 1i * original(:,2,:));     % From Jetson

            imageBlockRatio = 13; %how many blocks for a single image
            blockCount = size(noisy_Seq, 1);
            bitPerBlock = size(noisy_Seq, 2); %how many bits in one block, 5120
            uy_denoised = zeros(61470,1);
            %re-assemble the denoised sequences
            for k = 1:imageBlockRatio - 1
                uy_denoised((k-1)*bitPerBlock+1: k*bitPerBlock,1) = denoised_Seq(k,:).';
            end

            %padding
            % We only need the denoised sequence, we can use the noisy and
            % original RF from this Matlab script to compile the images
            pad = 30;
            tmp = (imageBlockRatio - 1) * bitPerBlock;
            uy_denoised(tmp + 1: tmp + pad) = denoised_Seq(imageBlockRatio, bitPerBlock - pad + 1:bitPerBlock).';

            %% Continue Image Conversion

            % For the Denoised Sequence
            uy2 = upfirdn(uy_denoised, h); %Apply RRC at the receiver side after CFO compensation
            dy = downsample(uy2,sps); %downsample uy2 to recover orginal x

            % QPSK Demodulation
            qpskDemod1 = comm.QPSKDemodulator('PhaseOffset', pi/4); % Same phase offset as the modulator
            demodulatedData = qpskDemod1(dy);

            % Compute and display the Symbol Error Rate (SER)

            %ser = sum(data ~= demodulatedData(1+filterSpan : numSymbols+filterSpan)) / numSymbols;
            ser = sum(Interleave_qpskSymbols ~= demodulatedData(1+filterSpan : numSymbols+filterSpan)) / numSymbols;

            % Scatter plot of the received signal prior to demodulation
            %figure;
            %scatterplot(dy);

            est_amp = sqrt(dy(1+filterSpan : numSymbols+filterSpan)'*dy(1+filterSpan : numSymbols+filterSpan)/numSymbols);
            signal_diff= dy(1+filterSpan : numSymbols+filterSpan)/est_amp - modulatedSignal(length(pilot)+1:numSymbols+length(pilot));
            MSE = signal_diff'*signal_diff/numSymbols;

            % disp(['total # of symbols received: ', num2str(numSymbols)]);
            % disp(['Chirp sequence length: ', num2str(chirp_len)]);
            %disp(['receive data at chunk index: ', num2str(i), '    Estimated CFO ', num2str(estimateOffset)]);
            %disp(['Sync. startIndex ', num2str(startIndex),'            Sync. peakvalue ', num2str(peakvalue)]);
            %disp(['MSE: ', num2str(MSE),'                     Symbol Error Rate: ', num2str(ser)]);

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



            %calculate two metrics PSNR and SSIM to compare the transmitted image and the received image.
            % The higher the PSNR, the better. An SSIM value of 1 is the best.
            [peaksnr, snr] = psnr(receivedImage, img);
            [ssimval, ssimmap] = ssim(receivedImage,img);
            %disp(['PSNR: ', num2str(peaksnr),'  SSIM value: ', num2str(ssimval)]);

            noisy_imgs(1,:,:,:) = imrotate(receivedImage,90);
            

            %% Send Image to Jetson
            %     if sending
            % Send images to Jetson
            save([fullFile,'\','orig_imgs.mat'],'orig_imgs','-v7.3');
            save([fullFile,'\','noisy_imgs.mat'],'noisy_imgs','-v7.3');

            url = 'http://192.168.1.10:5000';

            %'http://192.168.1.192:5000/unet' %house

            % File paths for the .mat files to upload
            file1_path = [fullFile,'\','orig_imgs.mat']; % Path to the first .mat file
            file2_path = [fullFile,'\','noisy_imgs.mat'];  % Path to the second .mat file

            % Construct the curl command for making a POST request with file upload
            command = sprintf(['curl -X POST %s/imageDenoise ' ... % Specify POST method
                '-F "file1=@%s" ' ... % Attach the first file as "file1"
                '-F "file2=@%s"'], url, file1_path, file2_path); % Attach the second file as "file2"

            % Execute the curl command
            [status, cmdout] = system(command); % Run the system command and capture the status and output

            % Check the status of the command execution
            if status == 0
                %disp('Response:'); % Display the server response if successful
                %disp(cmdout); % Print the response content
            else
                %disp('Error occurred:'); % Display an error message if the command fails
                %disp(cmdout); % Print the error details
            end
            %     end
            %
            %     break;
            % end %finish processing the chunk that contains the transmitted data
            %
            % if (i==numChunks)
            %     disp('Did not locate data, increase numChunks or decrease correlation threshold');
            % end
            
            

            fprintf('[4] Triggering Jetson display...\n');
            [status, cmdout] = system(sprintf('curl -X POST %s/finalize_images', url));
            fprintf('[DONE] Pipeline complete.\n');

            break;

        end


    end %finish processing all chunks

    if imageIndex > startIdx
        rem = loopTime - toc;
    else
        rem = firstLoopTime - toc;
    end
    disp(rem);
    pause(rem);
end


