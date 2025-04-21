function process(received_mat_path, output_dir,image_index)

    %raw_Img_Number_Offset = 70;
    cifarImage = load('data_batch_1.mat');
    data = cifarImage.data;
    load('Interleaver_dict_1000.mat');
    rng(15);

    imageBlockRatio = 13;
    pad = 30;
    numSymbols = 12288;
    sps = 5;
    rolloff = 0.25;
    filterSpan = 6;
    h = rcosdesign(rolloff, filterSpan, sps);
    qpskDemod = comm.QPSKDemodulator('PhaseOffset', pi/4);

    load(received_mat_path); % should define noised, denoised, original

    % Convert sequences
    noisy_Seq = squeeze(noised(:,1,:) + 1j * noised(:,2,:));
    denoised_Seq = squeeze(denoised(:,1,:) + 1j * denoised(:,2,:));
    original_Seq = squeeze(original(:,1,:) + 1j * original(:,2,:));

    blockCount = size(noisy_Seq, 1);
    bitPerBlock = size(noisy_Seq, 2);

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    perm_indices = Interleave_dict(image_index,:);
    %rawImg = data(image_index + raw_Img_Number_Offset, :);
    rawImg = data(image_index,:);
    img_gt = reshape(rawImg, [32, 32, 3]);
    img_gt = permute(uint8(img_gt), [2, 1, 3]);

    % Reassemble bitstreams
    uy_orig    = reassemble(original_Seq, 1, bitPerBlock, imageBlockRatio, pad);
    uy_noisy   = reassemble(noisy_Seq, 1, bitPerBlock, imageBlockRatio, pad);
    uy_denoised= reassemble(denoised_Seq, 1, bitPerBlock, imageBlockRatio, pad);

    % Demodulation
    dy_o = downsample(upfirdn(uy_orig, h), sps);
    dy_n = downsample(upfirdn(uy_noisy, h), sps);
    dy_d = downsample(upfirdn(uy_denoised, h), sps);

    % QPSK demod
    bits_o = bits_from_qpsk(qpskDemod, dy_o, perm_indices, numSymbols, filterSpan);
    bits_n = bits_from_qpsk(qpskDemod, dy_n, perm_indices, numSymbols, filterSpan);
    bits_d = bits_from_qpsk(qpskDemod, dy_d, perm_indices, numSymbols, filterSpan);

    % Reconstruct images
    img_o = reshape_image(bits_o);
    img_n = reshape_image(bits_n);
    img_d = reshape_image(bits_d);

    % Save images
    imwrite(img_o, fullfile(output_dir, sprintf('original-%d.png', 1)));
    imwrite(img_n, fullfile(output_dir, sprintf('noise-%d.png', 1)));
    imwrite(img_d, fullfile(output_dir, sprintf('denoise-%d.png', 1)));
end


function uy = reassemble(seq, idx, blockSize, blockRatio, pad)
    uy = [];

    for i = 1:blockRatio - 1
        uy = [uy; seq(i, :).'];
    end

    lastBits = seq(blockRatio, 5120 - pad + 1:5120).';
    uy = [uy; lastBits];
end

function bits = bits_from_qpsk(demod, signal, perm_indices, numSymbols, span)
    data = demod(signal);
    data = data(1+span : numSymbols+span);
    data_1(perm_indices) = data;
    bits = reshape(de2bi(data_1, 2, 'left-msb')', [], 1);
end

function img = reshape_image(bits)
    bytes = bi2de(reshape(bits, [], 8), 'left-msb');
    img = uint8(reshape(bytes, [32, 32, 3]));
end
