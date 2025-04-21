function send(orig_file,noisy_file,image_index)

server = 'http://192.168.1.10:5000';
%orig_file = '/Users/mac/Desktop/MATLAB/original_test_hardware_1.mat';
%noisy_file = '/Users/mac/Desktop/MATLAB/noisy_test_hardware_1.mat';
result_file = 'received_result.mat';
image_output_dir = 'output_images';

% Part1
fprintf('[1] Uploading input files...\n');
inference_flag = 'true';
cmd = sprintf(['curl -X POST %s/unet ' ...
    '-F "file1=@%s" -F "file2=@%s" -F "inference=%s" --output %s'], ...
    server, orig_file, noisy_file, inference_flag, result_file);


[status, response] = system(cmd);
if status ~= 0
    error('Failed to upload files or receive output: %s', response);
end
fprintf('[OK] Output saved to %s\n', result_file);

% Part2
fprintf('[2] Running local post-processing to generate images...\n');
if ~exist(image_output_dir,'dir')
    mkdir(image_output_dir);
end
process(result_file, image_output_dir,image_index);

% Part3
fprintf('[3] Uploading images to Jetson...\n');
for type = ["original", "noise", "denoise"]
    file = fullfile(image_output_dir, sprintf('%s-%d.png', type, 1));
    if isfile(file)
        curl_cmd = sprintf(['curl -X POST %s/upload_image ' ...
            '-F "image=@%s" -F "type=%s" -F "index=%d"'], ...
            server, file, type, 1);
        [~, resp] = system(curl_cmd);
        %fprintf('[%s-%d] %s\n', type, 1, strtrim(resp));
    else
        warning('Missing image: %s\n', file);
    end
end

end
