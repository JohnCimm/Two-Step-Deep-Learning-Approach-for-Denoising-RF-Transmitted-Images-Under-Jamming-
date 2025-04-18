%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Connor Nuibe
% L3Harris Senior Project - MIT RF Challenge
% Transmitter code for EMISignal1 from RF Challenge dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Pull Training Frames from "interpreted" file

% Extracting data is time consuming, ask user that way they can skip
% if signal is in memory
ans = input('Extract EMISignal1?\n y or n\n','s');

% loop through files (580 for EMISignal1) and extract each frames data
if  ans == "y"
    clear all;
    EMISig1 = [];
    lengths = zeros(580,1);
    for i = 0:579
        tic;
        % logic statement for naming convention
        if i < 10
            idx = "00" + i;
        elseif i < 100
            idx = "0" + i;
        elseif i > 100
            idx = num2str(i);
        end

        % file in "interpreted" file
        filename = "interpretedData/EMISignal1/EMISignal1_train_frame_0" + idx + "_.txt";
        fileId = fopen(filename,'r');

        % Pull character array of data
        [file cnt] = fscanf(fileId,'%s');
        fclose(fileId);

        % Build pattern structure from text file
        pattern = '\(\s*(-?\d*\.\d+|-?\d+)\s*([+-]?)\s*(-?\d*\.\d+|-?\d+)\s*j\)';
        matches = regexp(file,pattern,'tokens');

        % Initialize array
        complexNums = [];

        % loop through numbers
        for n = 1:length(matches)

            % Real part is the first match, imaginary is 3rd
            realPart = str2double(matches{n}{1});
            sign = matches{n}{2};
            imaginaryPart = str2double(matches{n}{3});

            % if number is missing assign it as 0
            if isempty(realPart)
                realPart = 0;
            end

            if isempty(imaginaryPart)
                imaginaryPart = 0;
            end

            if strcmp(sign,'-')
                imaginaryPart = -imaginaryPart;
            end

            % Build complex number
            complexNumber = realPart + j*imaginaryPart;
            complexNums = [complexNums,complexNumber];
        end

        % Concatenate with CommSig2 array
        EMISig1 = [EMISig1 complexNums];
        lengths(i+1) = length(complexNums);

        disp(['Done with frame ',idx]);
        disp(lengths(i+1));
        toc;
    end
end
% Create plot of traning frames built in
fs = 25e6;
freq = fft(EMISig1);

figure("Name","EMISig1 Time Domain");
t = linspace(0,length(EMISig1)/fs,length(EMISig1));
plot(t,real(EMISig1));
xlabel('time');
ylabel('amplitude');
title('Real Part of EMISig1');

figure("Name","EMISig1 Frequency Domain");
sigTransformed = fftshift(freq);
plot(abs(sigTransformed));
xlabel('Frequency');
ylabel('amplitude');
title('Frequency domain of EMISig1');
