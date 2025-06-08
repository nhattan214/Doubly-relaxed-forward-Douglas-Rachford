clc
clear all
close all


%% Create testing matrix

% Create 5 low-ranked submatrices
n1=300;
n2=400;
n3=100;
n4=200;
n5=100;

all_time=[];
all_iter=[];
all_re=[];

for t=1:20
t
A1= randn(n1,1)*randn(n1,1)';
A2= randn(n2,1)*randn(n2,1)';
A3= randn(n3,1)*randn(n3,1)';
A4=randn(n4,1)*randn(n4,1)';
A5=randn(n5,1)*randn(n5,1)';

%Groundtruth matrix
A=blkdiag(A1,A2,A3,A4,A5);
Rank_A=rank(A);

% Create Gaussian noise matrix

noise_mat= randn(size(A,1),size(A,2));



% Create mask matrix
mask = zeros(size(A,1),size(A,1)); 
samp_rate = 0.35;  % sampling rate  ~ samp_rate=0.15 means 15% of the entries are corrupted by noise 
chosen = randperm(size(A,1)*size(A,1),round(samp_rate*size(A,1)*size(A,1))); 
mask(chosen) = 1 ;

noise_mat=mask.*noise_mat;

A_noisy=A+noise_mat;

%% Algorithms
rho1=0.1;
rho2=0.1;

ell=2*rho2;
kappa=1;
alpha=1;

theta=1;
eta=1.4;
temp_delta=((eta*theta+2-2*theta)*alpha-(3*eta-2)*theta*ell)^2-8*(eta-2)*theta*kappa*(kappa+ell);

gamma_up=((eta*theta+2-2*theta)*alpha-(3*eta-2)*theta*ell+sqrt(temp_delta))/(4*theta*kappa*(kappa+ell)) -10^-20;
gamma_down=((eta*theta+2-2*theta)*alpha-(3*eta-2)*theta*ell-sqrt(temp_delta))/(4*theta*kappa*(kappa+ell)) -10^-20;

ub_eta=2+(2*kappa)/(theta*(kappa+ell));

lb_alpha=((3*eta-2)*theta*ell+2*sqrt(2*(eta-2)*theta*kappa*(kappa+ell)))/(eta*theta+2-2*theta)

pm.rho1 = rho1;
pm.rho2 = rho2;
pm.Ag = A;
pm.maxit=2000;
pm.reltol = 1e-6;
pm.A0=A_noisy;
pm.ratio=1;
pm.Rank_A=Rank_A;
pm.alpha=1;
pm.eta=1.4;
pm.theta=1;


pm2=pm;
pm2.alpha=0;

[X_GFDR,outputGFDR]=GFDR_LRSMC(pm,0);
[X_GFDR_0,outputGFDR_0]=GFDR_LRSMC(pm2,0);


time=[outputGFDR_0.time,outputGFDR.time];
err=[  outputGFDR_0.err(end),outputGFDR.err(end)];
iter=[size(outputGFDR_0.err,2),size(outputGFDR.err,2)];

all_time =[all_time; time ];
all_iter=[all_iter;iter];
all_re=[all_re;err];
end

mean_time=mean(all_time)
mean_iter=mean(all_iter)
mean_err=mean(all_re)

record=[mean_time,mean_iter,mean_err];

std_time=std(all_time)
std_iter=std(all_iter)
std_err=std(all_re)

