function [k,z,mse,omse]=DRFDR(data,x0,M,omega,n1,n2,r,m,gamma0,ratio,lambda,tol,maxit, eta, heuristic_on)
x=x0;
z=zeros(n1,n2);
y = z;

mse=[];
omse=[];
if heuristic_on==1
    gamma = ratio * gamma0;
else
    gamma =  gamma0;
end
k=0;
while(k<=maxit)
    
     yold = y;
     
 [ii,jj] = ind2sub([n1,n2], omega);

tempdata = gamma*(data - x(omega))/(1+gamma);
y = x + sparse(ii,jj,tempdata,n1,n2,m);
 temp = (2 - gamma*lambda)*y - x;
 [U,Sigma,V] = lansvd(temp,r,'L');
 z = U*Sigma*V';

 x = x + eta*(z-y);

  k=k+1;
 mse(k) = MSE(M,z);
 rel = norm(z(omega) - M(omega))/norm(M(omega));
 omse(k)=rel;
 
     if rel<tol
			break;
     end
  if heuristic_on==1
  if (norm(y - yold,'fro')/n1 >= 1e3/k || norm(y,'fro')/n1 > 1e10) && gamma > gamma0 % decrease gamma to guarantee convergence
    gamma = max(gamma/2,gamma0*0.9999);
  end
  else
    gamma=gamma;
  end

end
end

%%
function mse=MSE(a,b)
mse=norm(a-b,'fro')/norm(a,'fro');
end



































