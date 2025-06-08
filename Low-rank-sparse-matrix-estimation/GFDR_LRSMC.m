function [Y,output]=GFDR_LRSMC(pm, heuristic_on)



if isfield(pm,'maxit'); 
    maxit = pm.maxit; 
else 
    maxit = 5*N; % default value
end

if isfield(pm,'A0'); 
    A0 = pm.A0; 
else 
    A0 = zeros(N,1); % initial guess
end

if isfield(pm,'Ag'); 
    Ag = pm.Ag; 
else 
    Ag = zeros(N,1); % initial guess
end


if isfield(pm,'reltol'); 
    reltol = pm.reltol; 
else 
    reltol  = 1e-4; 
end

if isfield(pm,'theta'); 
    theta = pm.theta; 
else 
    theta  = 1; 
end

if isfield(pm,'rho1'); 
    rho1 = pm.rho1; 
else 
    rho1  = 1; 
end

if isfield(pm,'rho2'); 
    rho2 = pm.rho2; 
else 
    rho2  = 1; 
end

if isfield(pm,'eta'); 
    eta = pm.eta; 
else 
    eta  = 1; 
end

if isfield(pm,'alpha'); 
    alpha = pm.alpha; 
else 
    alpha  = 0; 
end


if isfield(pm,'ratio'); 
    ratio = pm.ratio; 
else 
    ratio  = 1; 
end

if isfield(pm,'Rank_A'); 
    Rank_A = pm.Rank_A; 
else 
    Rank_A  = 1; 
end

% Calculate gamma
n1=size(Ag,1);

ell=2*rho2;
kappa=1;
% alpha=1;


temp_delta=((eta*theta+2-2*theta)*alpha-(3*eta-2)*theta*ell)^2-8*(eta-2)*theta*kappa*(kappa+ell);

gamma0=((eta*theta+2-2*theta)*alpha-(3*eta-2)*theta*ell+sqrt(temp_delta))/(4*theta*kappa*(kappa+ell)) -10^-20


if heuristic_on==1
   
    gamma = ratio * gamma0;
else
    gamma =  gamma0;
end


X=A0;
% X=zeros(size(A0,1),size(A0,2));

Y=X;
Z = zeros(size(A0,1),size(A0,2));
% Main loop
total_time = 0;
eta
for it=1:maxit
     tic;
     Xold = X;
     % Update X
     X=(gamma/(gamma+1)).*(A0+Z);
     % Update Y
     subgrad=rho2*Subgrad_Cal(Y,Rank_A);
     grad=2*rho2*X;   

     temp_mat=(theta+1)*X-theta*Z-theta*gamma*grad+theta*gamma*subgrad;
     Yold=Y;
     Y=shrink(temp_mat,theta*gamma*rho1);

     Z=Z+eta.*(Y-X);

     time_iter = toc;

     rel = norm(Y-Yold,'fro')/norm(Yold,'fro');
     

    output.relerr(it)   = rel;

    output.err(it)      = norm(Y - Ag,'fro')/norm(Ag,'fro');

     if rel<reltol
			break;
     end
     total_time = total_time + time_iter;

     if heuristic_on==1
          if (norm(X - Xold,'fro')/n1 >= 1e3/it || norm(X,'fro')/n1 > 1e10) && gamma > gamma0 % decrease gamma to guarantee convergence
            gamma = max(gamma/2,gamma0*0.9999);
          end
     else
        gamma=gamma;
     end


end
output.time = total_time;

end

function subgrad=Subgrad_Cal(X,k)
[u,s,v]=svd(X);
s_vec=diag(s);
s_vec(k+1:size(s_vec,1))=0;
s=diag(s_vec);
subgrad=2*u*s*v';
end


