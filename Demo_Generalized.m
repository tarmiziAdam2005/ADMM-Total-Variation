
% Created on 2/12/2017 by Tarmizi Adam.
% Small demo for 1D-signal Total Variation (TV) denoising using the
% Alternating Direction Methods of Multiplier (ADMM).
% This demo file calls the function " ADMM_1D() " solver.

%This demo is to show NONCONVEX regularization for 1D total variation
% using the CAPPED L1 Penalty function with the GIST algorithm
% (General Iterative Shrinkage Algorithm) by Pinghua Gong.

% The function calls "ADMM_1D_CAPL1.m" function. Refer there  for more info

clc;
clear all;
close all;

%Load a piecewice defined function 
% You can use any  signal that you have
load testSig3.mat;
%load x.mat;
%load ecg3.mat;


y = testSig3;
%noisy_y = ecg3;
%y = x';

%add some noise to it
sigma = 5;
noisy_y = y + sigma * randn(1, length(y));


%% ********** parameter initialization*******
Nit = 100; % number of iterations
lam     = 0.0023; % Regularization parameter (play with this to see denoising effects)
rho     = 0.9; %penalty associated with the constraints (ADMM algorithm)
theta   = 0.09; % parameter related to capped L1 penalty (theta > 0) refer to paper.
%% ***********************************************************

%% ********** Run the TV-solver ***************

out = ADMM_1D_CAPL1(noisy_y, lam, rho, Nit, theta); %Run the Algorithm !!!

%% ********************************************

%%

rmse         = sqrt(mean((y'-out.sol).^2));

figure;
subplot(3,1,1)
plot(y);
axis tight;
title('Original Signal');

subplot(3,1,2);
plot(noisy_y)
axis tight;
title('Noisy Signal');

%figure;
subplot(3,1,3);
plot(out.sol);
axis tight;
title('TV Denoised');
