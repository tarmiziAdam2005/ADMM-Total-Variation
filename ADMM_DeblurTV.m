function out = ADMM_DeblurTV(f,Img,K,opts)
% Created on 12/12/2017 by Tarmizi Adam
% ADMM for deblurring
%
%The following code solves the following optimization problem
%
%         minimize_u lam/2||Ku - f ||_2^2 + ||Du||_1 + I(u)
%
% where I(u) is the indicator function (for box constraint).
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

%*** Variables for z and r subproblem ***
z           = v1;

%**** Lagrange multipliers ***
mu1         = zeros(row,col);
mu2         = mu1;
mu3         = mu1;

eigK        = psf2otf(K,[row col]); %In the fourier domain
eigKtK      = abs(eigK).^2;
Ktf = imfilter(f,K,'circular');

eigDtD      = abs(fft2([1 -1], row, col)).^2 + abs(fft2([1 -1]', row, col)).^2;

[D,Dt]      = defDDt(); %Declare forward finite difference operators
[Dux, Duy] = D(u);

    for k = 1:Nit
          
      %*** solve v - subproblem ***  
      x1          =  Dux + (1/rho)*mu1;
      x2          =  Duy + (1/rho)*mu2;
      
      v1 = shrink(x1,1/lam);
      v2 = shrink(x2,1/lam);
      
      %v1 = shrinkCapl1(x1,lam/rho,theta)
      %v2 = shrinkCapl1(x2,lam/rho,theta)
      
       %*** solve u - subproblem ***  
      u_old   = u;
      
      rhs     = lam*Ktf + Dt(rho*v1 - mu1,rho*v2 - mu2) + rho*z - mu3; 
      lhs     = lam*eigKtK + rho*eigDtD + rho;
      
      u       = fft2(rhs)./lhs;
      u       = real(ifft2(u));
     
       %*** solve z - subproblem ***
      z = min(255,max(u + mu3/rho,0));
      
      [Dux, Duy]  = D(u);
      
      mu1 = mu1 -rho*(v1- Dux);
      mu2 = mu2 -rho*(v2 - Duy);
      mu3 = mu3 - rho*(z - u);
      
      relError(k)    = norm(u - u_old,'fro')/norm(u, 'fro');
      
      if relError(k) < tol
          break;
      end
         
    end
    
    out.sol                 = u;
    out.relativeError       = relError(1:k);

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

function z = shrinkCapl1(x,r,theta)
    
    x1 = sign(x).*max(abs(x),theta);
    x2 = sign(x).*min(theta, max(abs(x)- r));
    
    if 0.5*(x1 - x).^2 + r*min(abs(x1),theta) <= 0.5*(x2 - x).^2 + r*min(abs(x2),theta)
            z = x1;
    else
            z = x2;
    end  
end

