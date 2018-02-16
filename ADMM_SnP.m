function out = ADMM_SnP(f,Img,K,opts)
% Created on 16/2/2018 by Tarmizi Adam
% Version 1.1
% ADMM for removing blur with salt and pepper noise
%
%The following code solves the following optimization problem
%
%         minimize_u lam*||Ku - f ||_1 + ||Du||_1 + I(u)
%
% where I(u) is the indicator function.
%%
%  Input:
%             f  : Noisy image (corrupted by salt and pepper noise)
%            Img : Original Image
%             K  : Point spread function (Convolution kernel)
%           opts : Options i.e., rho, regularization parameter, No of iterations etc.
%
%  Output   
%           out.sol      : Denoised image
%           out.relError : relative error 
%%

lam = opts.lam;  
tol = opts.tol; 
Nit = opts.Nit;

%******** Regularization parameter related to constraints ******
rho_v = 0.01;
rho_z = 0.1;
rho_r = 5;


relError        = zeros(Nit,1);
psnrGain        = relError;     % PSNR improvement every iteration
ssimGain        = relError;

[row, col]  = size(f);
u           = f;

%*** Variables for v subproblems ***
v1          = zeros(row,col);
%v2          = v1;

%*** Variables for z and r subproblem ***
z           = v1;
%r           = v1;

%**** Lagrange multipliers ***
mu_v1         = zeros(row,col);
mu_v2         = mu_v1;
mu_z         = mu_v1;
mu_r         = mu_v1;

eigK        = psf2otf(K,[row col]); %In the fourier domain
eigKtK      = abs(eigK).^2;
eigDtD      = abs(fft2([1 -1], row, col)).^2 + abs(fft2([1 -1]', row, col)).^2;

[D,Dt]      = defDDt(); %Declare forward finite difference operators
[Dux, Duy] = D(u);

lhs     = eigKtK + rho_v/rho_r*eigDtD + rho_z/rho_r; % From normal eqns.
q       = imfilter (u,K,'circular') -f;

tg = tic;
    for k = 1:Nit
        
      u_old   = u;
      
      %*** solve v - subproblem ***  
      x1          =  Dux + mu_v1/rho_v;
      x2          =  Duy + mu_v2/rho_v;
      
      v1 = shrink(x1,1/rho_v);
      v2 = shrink(x2,1/rho_v);
       
      %*** solve r - subproblem ***
      r = shrink(q + mu_r/rho_r, lam/rho_r);
        
      %*** solve u - subproblem ***  
      ftemp   = r + f -mu_r/rho_r;
      rhs     = imfilter(ftemp,K,'circular') + 1/rho_r*Dt(rho_v*v1 - mu_v1,rho_v*v2 - mu_v2) + rho_z/rho_r*z - mu_z/rho_r; 
      u       = fft2(rhs)./lhs;
      u       = real(ifft2(u));
      
      %*** solve z - subproblem ***
      z = min(255,max(u + mu_z/rho_z,0));
      
      [Dux, Duy]  = D(u);
     
      q   = imfilter(u,K,'circular') - f;
      
      %*** Update the Lagrange multipliers ***
      mu_v1 = mu_v1 -rho_v*(v1- Dux);
      mu_v2 = mu_v2 -rho_v*(v2 - Duy);
      
      mu_z = mu_z - rho_z*(z - u);
      mu_r = mu_r - rho_r*(r - q);
      
      %*** Some statistics, this might slow the cpu time of the algo. ***
      relError(k)    = norm(u - u_old,'fro')/norm(u, 'fro');
      %psnrGain(k)    = psnr_fun(u,double(Img));
      %ssimGain(k)    = ssim_index(u,double(Img));
      
      if relError(k) < tol
          break;
      end
           
    end
    
    tg = toc(tg);  

    out.sol                 = u;
    out.relativeError       = relError(1:k);
    out.psnrGain            = psnrGain(1:k);
    out.ssimGain            = ssimGain(1:k);
    out.cpuTime             = tg;
    out.psnrRes             = psnr_fun(u, double(Img));
    out.ssimRes             = ssim_index(u, double(Img));
    out.OverallItration     = size(out.relativeError,1);

end


function [D,Dt] = defDDt()
D  = @(U) ForwardDiff(U);
Dt = @(X,Y) Dive(X,Y);
end

function [Dux,Duy] = ForwardDiff(U)
 Dux = [diff(U,1,2), U(:,1,:) - U(:,end,:)];
 Duy = [diff(U,1,1); U(1,:,:) - U(end,:,:)];
end

function DtXY = Dive(X,Y)
  % Transpose of the forward finite difference operator
  % is the divergence fo the forward finite difference operator
  DtXY = [X(:,end) - X(:, 1), -diff(X,1,2)];
  DtXY = DtXY + [Y(end,:) - Y(1, :); -diff(Y,1,1)];   
end

function z = shrink(x,r)
z = sign(x).*max(abs(x)-r,0);
end

