clc
clear all
close all


img_index=3;  
size_image=100;

threshold=0.1;
switch img_index
           
    case 1
        P=imread('cameraman.tif');
        P=imresize(P,[size_image,size_image]);
        P=double(P)./255;
        P_reshape=reshape(P,[],1);
        P_dct=dct(P_reshape,"Type",2);
        index=find(abs(P_dct)<threshold);
        P_dct_filtered=P_dct;
        P_dct_filtered(index)=0;
        recon=idct(P_dct_filtered)*255;
        recon=reshape(recon,size_image,size_image);
        fprintf('Relative error: %f',norm(recon-P,'fro')/norm(P,'fro'));
        recon=uint8(recon);
  
        
    case 2
        P=imread('mri.tif');
        P=imresize(P,[size_image,size_image]);
        P=double(P)./255;
        P_reshape=reshape(P,[],1);
        P_dct=dct(P_reshape,"Type",2);
        index=find(abs(P_dct)<threshold);
        P_dct_filtered=P_dct;
        P_dct_filtered(index)=0;
        recon=idct(P_dct_filtered)*255;
        recon=reshape(recon,size_image,size_image);
        fprintf('Relative error: %f',norm(recon-P,'fro')/norm(P,'fro'));
        recon=uint8(recon);
   

    otherwise
        P=imread('moon.tif');
        P=imresize(P,[size_image,size_image]);
        P=double(P)./255;
        P_reshape=reshape(P,[],1);
        P_dct=dct(P_reshape,"Type",2);
        index=find(abs(P_dct)<0.1);
        P_dct_filtered=P_dct;
        P_dct_filtered(index)=0;
        recon=idct(P_dct_filtered).*255;
        recon=reshape(recon,size_image,size_image);
        fprintf('Relative error: %f',norm(recon-P,'fro')/norm(P,'fro'));
        recon=uint8(recon);
  
  
end

compressed=reshape(double(recon),[],1)/255;
x=compressed;
N=size(P_dct,1);

DCT_basis=dct(eye(N));
IDCT_basis=idct(eye(N));

x_dct=P_dct_filtered;
miss_rate=[0.1,0.15,0.20,0.25];
for r=1:1
fprintf('Missing ratio -%0.2f--\n',r);
missing_ratio=miss_rate(r);

all_re=[];
all_time=[];
all_iter=[];

for time=1:1 

Phi = eye(N);
missing = randperm(N, ceil(N*missing_ratio));
Phi(missing,:)=[];
x_plot=x;
x_plot(missing)=0;

b=Phi*x;   % Measurements
x0 = zeros(N,1)+0.01;
A=Phi*IDCT_basis;
x_plot=reshape(x_plot,size_image,size_image)*255;
x_plot=uint8(x_plot);
%%
lambda=0.0001;
x_ref=x_dct;

pm.lambda = lambda;
pm.delta = pm.lambda*100;
pm.xg = x_ref;
pm.maxit=3000;
pm.reltol = 1e-5;
pm.ell=norm(A*A');

pm_New_TOS=pm;
pm_New_TOS.eta=1.8;
pm_New_TOS.ratio=200;
pm_New_TOS.Theta=1;

fprintf('Running DRFDR ...\n');
[x_DRFDR,outputDRFDR] = New_TOS(A,b,pm_New_TOS, 1);

reconstructed_DRFDR=reshape(idct(x_DRFDR),size_image,size_image)*255;
reconstructed_DRFDR=uint8(reconstructed_DRFDR);

RE_all=[RE_Cal(double(reconstructed_DRFDR),double(recon))];
time_all=[outputDRFDR.time];
% fprintf('Relative')
iter_all=[size(outputDRFDR.err,2)];

all_iter=[all_iter;iter_all];
all_re=[all_re;RE_all];
all_time=[all_time;time_all];

end
ALL_MEAN=[mean(all_time),mean(all_iter),mean(all_re)];
Results(r).All_ave=ALL_MEAN;
end

figure;
subplot(1,3,1)
imshow(recon)
title('Ground truth image')
subplot(1,3,2)
imshow(x_plot)
title('Observed image')
subplot(1,3,3)
imshow(reconstructed_DRFDR)
title('Reconstructed image')
axis tight
%%
function RE=RE_Cal(recon,truth)
RE=norm(recon-truth,'fro')/norm(truth,'fro');
end


