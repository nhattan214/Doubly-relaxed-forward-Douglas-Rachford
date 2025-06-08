clc
clear all
close all

M=xlsread('Data_new.xlsx',3);
M=Preprocess(M,20);
n1=size(M,1);
n2=size(M,2);
r=rank(M);

max_run=30;

RUN_TIME=[];
RUN_ITER=[];
RUN_ERR=[];


for time=1:max_run
    fprintf('Time--%d',time)
samp_rate=0.6; %This is R
df = r*(n1+n2-r);
m = round(min(1000*df,samp_rate*n1*n2));

Omega = sort(randsample(n1*n2,m)); %%sample index
ssigma = 0;
data = M(Omega) + ssigma*randn(size(Omega));

p  = m/(n1*n2)
maxit = 2000; 
tol = 1e-4;
[i, j] = ind2sub([n1,n2], Omega);
x0 = sparse(i,j,data,n1,n2,m);
mse0=MSE(M,x0);
omse0=norm(x0(Omega) - M(Omega))/norm(M(Omega));   


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%-----DRFDR-----
%%%%%%%%%%%%%%%%%%%%%%%%
lambda = 1.8e-6;
ratio = 1e1;
gamma0 = 0.22;
eta=1.8;
tim=clock;
[k,x,mse,omse]=DRFDR(data,x0,M,Omega,n1,n2,r,m,gamma0,ratio,lambda,tol,maxit,eta,1);
tim=etime(clock,tim);
it=k;
ee=MSE(M,x);


all_time=[tim]
all_iter=[it]
all_error=[ee]


RUN_TIME=[RUN_TIME; all_time];
RUN_ITER=[RUN_ITER;all_iter];
RUN_ERR=[RUN_ERR;all_error];

end

MEAN_TIME=mean(RUN_TIME);
MEAN_ITER=mean(RUN_ITER);
MEAN_ERR=mean(RUN_ERR);



%%

function R=Preprocess(M,desired_rank)
%Create a rank-20 approximation
[U,S,V]= svd(M);
U=U(:,1:size(S,1));
S=diag(diag(S));
R=U(:,1:desired_rank)*S(1:desired_rank,1:desired_rank)*V(:,1:desired_rank)';
end

function mse=MSE(a,b)
mse=norm(a-b,'fro')/norm(a,'fro');
end