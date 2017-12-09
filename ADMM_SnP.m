function out = ADMM_SnP(f,Img,K,opts)
% Created on 5/12/2017 by Tarmizi Adam
% ADMM for removing blur with salt and pepper noise
%
%The following code solves the following optimization problem
%
%         minimize_u lam/2||Ku - f ||_1 + ||Du||_1 + I(u)
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
rho = opts.rho; 
tol = opts.tol; 
Nit = opts.Nit;
rho_r = opts.rho_r;

theta= 0.09;

relError        = zeros(Nit,1);

[row, col]  = size(f);
u           = f;

%*** Variables for v subproblems ***
v1          = zeros(row,col);
v2          = v1;

%*** Variables for z and r subproblem ***
z           = v1;
r           = v1;

%**** Lagrange multipliers ***
mu1         = zeros(row,col);
mu2         = mu1;
mu3         = mu1;
mu4         = mu1;

eigK        = psf2otf(K,[row col]); %In the fourier domain
eigKtK      = abs(eigK).^2;
eigDtD      = abs(fft2([1 -1], row, col)).^2 + abs(fft2([1 -1]', row, col)).^2;

[D,Dt]      = defDDt(); %Declare forward finite difference operators
[Dux, Duy] = D(u);

Ktf = imfilter(f,K,'circular');
q   = imfilter (u,K,'circular') -f;
    for k = 1:Nit
        
        
      %*** solve v - subproblem ***  
      x1          =  Dux + (1/rho)*mu1;
      x2          =  Duy + (1/rho)*mu2;
      
      v1 = shrink(x1,lam/rho);
      v2 = shrink(x2,lam/rho);
      
       %*** solve r -  subproblem ***
      r = shrink(q + mu4/rho_r, lam/rho_r);
        
       %*** solve u - subproblem ***  
      u_old   = u;
      rhs     = rho*Ktf + imfilter(rho_r*r-mu4,K,'circular') + Dt(rho*v1 - mu1,rho*v2 - mu2) + rho*z - mu3; 
      lhs     = rho*eigKtK + rho*eigDtD + rho;
      
      u       = fft2(rhs)./lhs;
      u       = real(ifft2(u));
      
      
       %*** solve z - subproblem ***
      z = min(255,max(u + mu3/rho,0));
      
      [Dux, Duy]  = D(u);
     
      q   = imfilter (u,K,'circular') - f;
      
      mu1 = mu1 -rho*(v1- Dux);
      mu2 = mu2 -rho*(v2 - Duy);
      mu3 = mu3 - rho*(z - u);
      %mu4 = mu4 + (r - imfilter(u,K,'circular')+ f);
      mu4 = mu4 - rho_r*(r - q);
      
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
