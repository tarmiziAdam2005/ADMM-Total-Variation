% Demo to clean salt and pepper noise using Total variation (TV)
% denoising. Algorithm used is ADMM.
% 
% This script calls the main solver "ADMM_SnP.m" to solve the
% denoising problem. Refer to "ADMM_SnP.m" for more details
%
% Created on 9/12/2017 by Tarmizi adam, tarmizi_adam2005@yahoo.com

clc;
clear all;
close all;


Img = imread('Parrots.tif');

if size(Img,3) > 1
    Img = rgb2gray(Img);
end

K     =   fspecial('average',1); % For denoising
f = imfilter(Img,K,'circular');

f = imnoise(f,'salt & pepper',0.08);

f = double(f);


opts.lam = 180.0;
opts.rho = 2;
opts.rho_r = 0.5;
opts.tol = 1e-3;
opts.Nit = 400;

out = ADMM_SnP(f,Img,K,opts);

figure;
imshow(uint8(f));

figure;
imshow(uint8(out.sol));

figure;
imshow(Img);
