clear all; clc; close all
n1 =5000; n2 = 5000; r = 10;
max_run=2;

RUN_TIME=[];
RUN_ITER=[];
RUN_ERR=[];

for time=1:max_run
M = randn(n1,r)*randn(r,n2);
df = r*(n1+n2-r);
%Choose rate as 0.1 or 0.15
m = round(min(1000*df,0.15*n1*n2));
Omega = sort(randsample(n1*n2,m)); %%sample index
ssigma = 0;
data = M(Omega) + ssigma*randn(size(Omega));
%lambda = 1e-6;
%%%%%%%%%%%%%%%%%%%%%%%%%%%
p  = m/(n1*n2)
maxit = 1000; 
tol = 1e-4
[i, j] = ind2sub([n1,n2], Omega);
x0 = sparse(i,j,data,n1,n2,m);
mse0=MSE(M,x0);
omse0=norm(x0(Omega) - M(Omega))/norm(M(Omega));   


%%%-----DRFDR------
%%%%%%%%%%%%%%%%%%%%%%%%

lambda = 1.8e-6;
ratio = 1e6;
gamma0 = 0.2;
eta=1.8;
tim=clock;
[k,x,mse,omse]=DRFDR(data,x0,M,Omega,n1,n2,r,m,gamma0,ratio,lambda,tol,maxit,eta,1);
tim=etime(clock,tim)
it=k
ee=MSE(M,x)




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

R=[MEAN_TIME,MEAN_ITER,MEAN_ERR];


%%
function mse=MSE(a,b)
mse=norm(a-b,'fro')/norm(a,'fro');
end


