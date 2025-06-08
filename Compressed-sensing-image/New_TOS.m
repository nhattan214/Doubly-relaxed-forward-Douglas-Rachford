function [y, output] = New_TOS(A,b,pm, heuristic_on)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%         min_x .5||Ax-b||^2 + lambda(|x|_1- alpha |x|_2)             %%%
%%%    phi = 0.5||Ax-b||^2; g = lambda |x|_1; h = -alpha*lambda*|x|_2   %%%
%%%                                                                     %%%
%%% Input: dictionary A, data b, parameters set pm                      %%%
%%%        pm.lambda: regularization paramter                           %%%
%%%                                 %%%
%%%        pm.maxit: max iterations                                     %%%
%%%        pm.reltol: rel tolerance for DRDC: default value: 1e-6       %%%
%%%        pm.alpha: alpha in the regularization,default value: 1       %%%
%%% Output: computed coefficients z (shrinkage operator result)         %%%
%%%        output.relerr: relative error of yold and y                  %%%
%%%        output.obj: objective function of x_n:                       %%%
%%%        obj(x) = lambda(|x|_1 - alpha |x|_2)+0.5|Ax-b|^2             %%%
%%%        output.res: residual of x_n: norm(Ax-b)/norm(b)              %%%
%%%        output.err: error to the ground-truth: norm(z-xg)/norm(xg)   %%%
%%%        output.time: computational time                              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[M,N]       = size(A); 
% start_time  = tic; 

%% parameters
if isfield(pm,'lambda'); 
    lambda = pm.lambda; 
else
    lambda = 1e-5;  % default value
end
% parameter for ADMM
if isfield(pm,'Theta'); 
    Theta = pm.Theta; 
else
    Theta=1;
end


%%

% maximum number of iterations
if isfield(pm,'maxit'); 
    maxit = pm.maxit; 
else 
    maxit = 5*N; % default value
end
% initial guess
if isfield(pm,'x0'); 
    x0 = pm.x0; 
else 
    x0 = zeros(N,1); % initial guess
end
if isfield(pm,'xg'); 
    xg = pm.xg; 
else 
    xg = x0;
end
if isfield(pm,'reltol'); 
    reltol = pm.reltol; 
else 
    reltol  = 1e-6; 
end
if isfield(pm,'alpha'); 
    alpha = pm.alpha;
else 
    alpha = 1;
end

if isfield(pm,'r'); 
    r_param = pm.r;
else 
    r_param = 1;
end

if isfield(pm,'eta'); 
    eta = pm.eta; 
else
    eta = 1;
end

if isfield(pm,'beta0'); 
    beta_0 = pm.beta0;
else 
    beta_0 = 1;
end

if isfield(pm,'ratio'); 
    ratio = pm.ratio;
else 
    ratio = 1;
end

if isfield(pm,'ell');
    ell = pm.ell;
else
    ell = 1;
end

if isfield(pm,'k_size'); 
%     fprintf('using k(n) = %.2f\n',opts.k_size);
    k_size_list = ones(maxit,1)*pm.k_size;
else 
%     fprintf('using k(n) = (1+1/n)^2\n')
    k_size_list = @(n) (1+1/n)^2;
end

%% initialize
x 		= x0; 
y 		= x0;
z       = x0;

% ell=norm(A*A');

if heuristic_on==1
    % beta = beta_0*ratio; 
    % min(1/ell,(sqrt(8*(2-eta)*ell^2)/(4*ell^2)))-(10^-6)
    beta = ratio*((sqrt(8*(2-eta)*ell^2)/(4*ell^2))-10^-6)
    % fprintf('beta*lambda New_TOS %d',beta*lambda);
else
    beta=((sqrt(8*(2-eta)*ell^2)/(4*ell^2))-(10^-6))
    % fprintf('beta NewTOS = %d',beta);
end

% beta=0.1;

obj         = @(x) .5*norm(A*x-b)^2 + lambda*(norm(x,1)-alpha*norm(x));
output.pm   = pm;

% method 1
% D_inv = inv(beta*A'*A + eye(N));
% method 2
L = chol(speye(M) + beta*(A*A'), 'lower');
L = sparse(L);
U = sparse(L');

Atb   = A'*b;


total_time = 0;


for it = 1:maxit
    
    tic;
    % select k size
    k_size = k_size_list(it);
    q = beta*Atb + z;

    % method 2
    
    xn = q - beta*(A'*(U \ ( L \ (A*q) )));
    
    % Subgradient
    if norm(y) ~=0
        sub_grad = y/norm(y);
    else
        sub_grad = 0;
    end
    
    yn = shrink(2*xn-z+beta*lambda*sub_grad,lambda*beta);

    zn = z +  eta*(yn - xn);
    time_iter = toc;
  
    relerr      = norm(yn - y)/norm(y);
    residual    = norm(A*zn - b)/norm(b);

    if heuristic_on ==1
    if (norm(xn - x) >= 1e3/it || norm(x) > 1e10) && beta > beta_0 % decrease gamma to guarantee convergence
       beta = max(beta/2,beta_0*0.9999);
    end
    else
       beta=beta;
    end

    x = xn;
    y = yn;
    z = zn;
    
    output.relerr(it)   = relerr;
    output.obj(it)      = obj(z);
    output.res(it)      = residual;
    output.err(it)      = norm(y - xg)/norm(xg);
    total_time = total_time + time_iter;
    if relerr < reltol && it > 2  
        break;
    end

end
output.time = total_time;
end


function z = shrink(x, r)
    z = sign(x).*max(abs(x)-r,0);
end

