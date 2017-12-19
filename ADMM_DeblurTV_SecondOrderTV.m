function out = ADMM_DeblurTV_SecondOrderTV(f,Img,K,opts)
% Created on 19/12/2017 by Tarmizi Adam
% ADMM second order Total Variation image deblurring/denoising
%
%The following code solves the following optimization problem
%
%         minimize_u lam/2||Ku - f ||_2^2 + ||DDu||_1 + I(u)
%
% where I(u) is the indicator function (for box constraint).
% This could be seen as the LLT (Lysaker-Lundervold-Tai) model 
% with and added indicator function (a box constraint). The one 
% without the indicator function (original LLT model) is 
% "HOTV.m" file. However, this % script seems more robust because of the added indicator  
% function.
%%
%  Input:
%             f  : Noisy & Blurred image (corrupted by additive Gaussian noise)
%            Img : Original Image
%             K  : Point spread function (Convolution kernel)
%           opts : Options i.e., rho, regularization parameter, No of iterations etc.
%
%  Output   
%           out.sol      : Denoised image
%           out.relError : relative error 
%%

lam = opts.lam; 
rho = opts.rho; 
tol = opts.tol; 
Nit = opts.Nit;

%theta= 0.09;

relError        = zeros(Nit,1);

[row, col]  = size(f);
u           = f;

%*** Variables for v subproblems ***
v1          = zeros(row,col);
v2          = v1;
v3          = v1;
v4          = v1; 

%*** Variables for z and r subproblem ***
z           = v1;

%**** Lagrange multipliers ***
mu1         = zeros(row,col);
mu2         = mu1;
mu3         = mu1;
mu4         = mu1;
mu5         = mu1; % multiplier related to z-subproblem (projection)

eigK        = psf2otf(K,[row col]); %In the fourier domain
eigKtK      = abs(eigK).^2;
Ktf = imfilter(f,K,'circular');

eigDDtDD = abs(psf2otf([1 -2 1],[row col])).^2 + abs(psf2otf([1 -1;-1 1],[row col])).^2 ...
            + abs(psf2otf([1 -1;-1 1],[row col])).^2 + abs(psf2otf([1;-2;1],[row col])).^2;

[DD,DDt] = defDDt2(); %Declare forward finite difference operators
[Duxx,Duxy,Duyx,Duyy] = DD(u);

    for k = 1:Nit
          
      %*** solve v - subproblem ***  
      x1          =  Duxx + (1/rho)*mu1;
      x2          =  Duxy + (1/rho)*mu2;
      x3          =  Duyx + (1/rho)*mu3;
      x4          =  Duyy + (1/rho)*mu4;
      
      v1 = shrink(x1,1/lam);
      v2 = shrink(x2,1/lam);
      v3 = shrink(x3,1/lam);
      v4 = shrink(x4,1/lam);
      
      %v1 = shrinkCapl1(x1,lam/rho,theta)
      %v2 = shrinkCapl1(x2,lam/rho,theta)
      
       %*** solve u - subproblem ***  
      u_old   = u;
      
      rhs     = lam*Ktf + DDt(rho*v1 - mu1,rho*v2 - mu2,rho*v3 - mu3,rho*v4 - mu4) + rho*z - mu5; 
      lhs     = lam*eigKtK + rho*eigDDtDD + rho;
      
      u       = fft2(rhs)./lhs;
      u       = real(ifft2(u));
     
       %*** solve z - subproblem ***
      z = min(255,max(u + mu5/rho,0));
      
      [Duxx,Duxy,Duyx,Duyy] = DD(u);
      
      mu1 = mu1 -rho*(v1- Duxx);
      mu2 = mu2 -rho*(v2 - Duxy);
      mu3 = mu3 -rho*(v3 - Duyx);
      mu4 = mu4 -rho*(v4 - Duyy);
      
      mu5 = mu5 - rho*(z - u);
      
      relError(k)    = norm(u - u_old,'fro')/norm(u, 'fro');
      
      if relError(k) < tol
          break;
      end
         
    end
    
    out.sol                 = u;
    out.relativeError       = relError(1:k);

end


function [DD,DDt] = defDDt2
        % defines finite difference operator D
        % and its transpose operator
        DD  = @(U) ForwardD2(U);
        DDt = @(Duxx,Duxy,Duyx,Duyy) Dive2(Duxx,Duxy,Duyx,Duyy);
 end

function [Duxx Duxy Duyx Duyy] = ForwardD2(U)
        %
        Duxx = [U(:,end) - 2*U(:,1) + U(:,2), diff(U,2,2), U(:,end-1) - 2*U(:,end) + U(:,1)];
        Duyy = [U(end,:) - 2*U(1,:) + U(2,:); diff(U,2,1); U(end-1,:) - 2*U(end,:) + U(1,:)];
        %
        Aforward = U(1:end-1, 1:end-1) - U(  2:end,1:end-1) - U(1:end-1,2:end) + U(2:end,2:end);
        Bforward = U(    end, 1:end-1) - U(      1,1:end-1) - U(    end,2:end) + U(    1,2:end);
        Cforward = U(1:end-1,     end) - U(1:end-1,      1) - U(  2:end,  end) + U(2:end,    1);
        Dforward = U(    end,     end) - U(      1,    end) - U(    end,    1) + U(    1,    1);
        % 
        Eforward = [Aforward ; Bforward]; Fforward = [Cforward ; Dforward];
        Duxy = [Eforward, Fforward]; Duyx = Duxy;
        %
  end

  function Dt2XY = Dive2(Duxx,Duxy,Duyx,Duyy)
        %
        Dt2XY =         [Duxx(:,end) - 2*Duxx(:,1) + Duxx(:,2), diff(Duxx,2,2), Duxx(:,end-1) - 2*Duxx(:,end) + Duxx(:,1)]; % xx
        Dt2XY = Dt2XY + [Duyy(end,:) - 2*Duyy(1,:) + Duyy(2,:); diff(Duyy,2,1); Duyy(end-1,:) - 2*Duyy(end,:) + Duyy(1,:)]; % yy
        %
        Axy = Duxy(1    ,    1) - Duxy(      1,    end) - Duxy(    end,    1) + Duxy(    end,    end);
        Bxy = Duxy(1    ,2:end) - Duxy(      1,1:end-1) - Duxy(    end,2:end) + Duxy(    end,1:end-1);
        Cxy = Duxy(2:end,    1) - Duxy(1:end-1,      1) - Duxy(  2:end,  end) + Duxy(1:end-1,    end);
        Dxy = Duxy(2:end,2:end) - Duxy(  2:end,1:end-1) - Duxy(1:end-1,2:end) + Duxy(1:end-1,1:end-1);
        Exy = [Axy, Bxy]; Fxy = [Cxy, Dxy];
        %
        Dt2XY = Dt2XY + [Exy; Fxy];
        Dt2XY = Dt2XY + [Exy; Fxy];
   end

function z = shrink(x,r)
z = sign(x).*max(abs(x)-r,0);
end

function z = shrinkCapl1(x,r,theta)
    
    x1 = sign(x).*max(abs(x),theta);
    x2 = sign(x).*min(theta, max(abs(x)- r));
    
    if 0.5*(x1 - x).^2 + r*min(abs(x1),theta) <= 0.5*(x2 - x).^2 + r*min(abs(x2),theta)
            z = x1;
    else
            z = x2;
    end  
end

