function distorted_img = distortion_generator( img, dist_type, level, seed )
    %% set distortion parameter
    gblur_level = [7,15,39,91,199];
    wn_level = [-10,-7.5,-5.5,-3.5,0];
    jpeg_level = [43,12,7,4,0];
    jp2k_level = [0.46,0.16,0.07,0.04,0.02]; % bit per pixel
    motion_level = [1,2,3,4,5];
    pink_level = [0.3,0.7,1.2,2,4];
    dither_level = [64,32,16,8,4];
    dark_level = [1.5,3,4.5,6,7.5];
    bright_level = [1.5,3,4.5,6,7.5];
    
    map = 1;
    %% distortion generation
    switch dist_type
        case 1
            hsize = gblur_level(level);
            h = fspecial('gaussian', hsize, hsize/6);
            distorted_img = imfilter(img,h,'symmetric');
        case 2
            rng(seed);
            distorted_img = imnoise(img,'gaussian',0,2^(wn_level(level)));
        case 3
            testName = [num2str(randi(intmax)) '.jpg'];
            imwrite(img,testName,'jpg','quality',jpeg_level(level));
            distorted_img = imread(testName);
            delete(testName);
        case 4
            testName = [num2str(randi(intmax)) '.jp2'];
            imwrite(img,testName,'jp2','CompressionRatio', 24 / jp2k_level(level));
            distorted_img = imread(testName);
            delete(testName);
        case 5 %strech contrast
            switch level
                case 1
                    distorted_img = imadjust(img,stretchlim(img),[0.2 0.8]);
                case 2
                    distorted_img = imadjust(img,stretchlim(img),[0.3 0.7]);
                case 3
                    distorted_img = imadjust(img,stretchlim(img),[0.4 0.6]);
                case 4
                    distorted_img = imadjust(img,stretchlim(img),[0.45 0.55]);
                case 5
                    distorted_img = imadjust(img,stretchlim(img),[0.49 0.51]);
            end
            
        case 6 %pink noise        
            h = size(img,1);
            w = size(img,2);    
            
            fnoise_R = randnd(-1,2^9);fnoise_G = randnd(-1,2^9);fnoise_B = randnd(-1,2^9);
            fnoise_R = fnoise_R/max(abs(fnoise_R(:)));
            fnoise_G = fnoise_G/max(abs(fnoise_G(:)));
            fnoise_B = fnoise_B/max(abs(fnoise_B(:)));
            fnoise2 = zeros(512,512,3);
            fnoise2(:,:,1) = fnoise_R;fnoise2(:,:,2) = fnoise_G;fnoise2(:,:,3) = fnoise_B;
            fnoise2 = imresize(fnoise2,[h w]);
            fnoise2 = fnoise2*255;
%             
            
            wei = pink_level(level);
            
            distorted_img = double(img) + wei*fnoise2;
            distorted_img = floor(distorted_img);
            distorted_img = uint8(distorted_img);
         case 7 %dither
             dither = dither_level(level);
             [idx,map] = rgb2ind(img,dither);
             distorted_img = uint8(ind2rgb(idx,map) * 255);
         case 8 %overexposure
            g = bright_level(level);
%             distorted_img = imadjust(img,[],[],[g g g]);
            img = double(img);
            img = img/255;
            distorted_img = min(1.0, img + 0.1*g);  % Shifted
            distorted_img = round(255*distorted_img);
            distorted_img = uint8(distorted_img);
         case 9 %underexposure
            g = dark_level(level);
            img = double(img);
            img = img/255;
            distorted_img = min(1.0, img - 0.1*g);  % Shifted
            distorted_img = round(255*distorted_img);
            distorted_img = uint8(distorted_img);
%             distorted_img = imadjust(img,[],[],[g g g]);
    end
end

