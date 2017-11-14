function rgb_result=saturate(rgb,factor)
%Saturates of Desaturates an image
%   J=saturate(I,factor) returns an image J that is factor times the
%   saturation of 8bit color image I. So 0 is no saturation, 2 is double
%   andsoforth. You get the idea. Otherwise visit me at: www.timzaman.nl
%
%   Examples
%   --------
%   Removes all saturation
%
%       I = imread('rice.png');
%       J = saturate(I,0);
%       figure, imshow(I), figure, imshow(J)
%
%   Doubles the saturation
%
%       I = imread('rice.png');
%       J = saturate(I,2);
%       figure, imshow(I), figure, imshow(J)
%
%
%   Note
%   ----
%   This is the first version without much support
%
%   Class Support
%   -------------
%   The input image rgb should be 8bit unsigned
%
%   See also makecform, applycform
%   Written by Tim Zaman, TU Delft, 2011
%   This work, unless otherwise expressly stated, is licensed under a
%   Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
%   Obviously provided "AS-IS"
 
rgblab=makecform('srgb2lab');
labrgb=makecform('lab2srgb');
lab = applycform(rgb,rgblab);
lab=int16(lab);
lab=lab-128;
lab(:,:,2)=lab(:,:,2)*factor;
lab(:,:,3)=lab(:,:,3)*factor;
lab=uint8(lab+128);
rgb_result = applycform(lab,labrgb);
 
end