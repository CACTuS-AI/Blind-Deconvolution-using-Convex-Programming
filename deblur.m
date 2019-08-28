%% Blind Deconvolution using Convex Programming---Image Deblurring 
%% Figure 6,7,8
%% Ali Ahmed 

clear all;
close all;
%% Path 
addpath(fullfile('minFunc'));
addpath(fullfile('minFunc_2012'));
addpath(fullfile('minFunc','compiled'));
addpath(fullfile('minFunc','mex'));

%% Load Data

rgb = double(imread('shapes.png'));
x = mean(rgb,3);
x = double(x)/norm(x,'fro');
imagesc(x); title('Original shapes image'),colormap(gray),colorbar;
L1 = size(x,1); 
L2 = size(x,2); 
L = L1*L2; 
%% Useful functions

mat = @(x) reshape(x,L1,L2);
vec = @(x) x(:);

%% Blur Kernel

blur_kernel = fspecial('motion',30,45);
% h1 = fspecial('disk',7);
% h1 = fspecial('gaussian',[20,20],5)
[K1 K2] = size(blur_kernel); 
blur_kernel = blur_kernel/norm(blur_kernel,'fro');
w = zeros(L1,L2);
w(L1/2-(K1+1)/2+2:L1/2+(K1+1)/2,L2/2-(K2+1)/2+2:L2/2+(K2+1)/2) = blur_kernel; % K1 and K2 are odd; change if K1 and K2 even

%% Computing matrix B; see blind deconvolution using convex programming paper for notations 

w_vec = vec(w);
Indw = zeros(L,1);
Indw = abs(w_vec)>0;
figure;
plot(Indw);

j = 1;
K = sum(Indw);
B = sparse(L,K); 
h = zeros(K,1);
for i = 1:L
    if(Indw(i) == 1)
        B(i,j) = Indw(i);
        h(j) =w_vec(i);
        j = j+1;
    end
end

%% Define function BB

BB = @(x) mat(B*x);
BBT = @(x) B'*vec(x);
w1 = BB(h);   % w = Bh
figure;
imagesc(mat(w1)),title('blur kernel'), colormap(gray), colorbar;

%% 2D convolution

figure;
conv_wx = ifft2(fft2(x).*fft2(BB(h)));
conv_wx_image = fftshift(mat(conv_wx));
figure;
imagesc(conv_wx_image),title('Convolution of original image with blur kernel'), colormap(gray),colorbar;


%% Compute and display wavelet coefficients of the original and blurred image

[alpha_conv,l] = wavedec2(conv_wx_image,4,'db1');
figure;
plot(alpha_conv); title('wavelet coefficients of the convolved image');

[alpha_x,l] = wavedec2(x,4,'db1');
figure;
plot(alpha_x); title('wavelet coefficients of the original image');

%% C selected by wavelet coeffs of blurred\original\both image

alpha = alpha_x;
Ind = zeros(1,length(alpha));
Ind_alpha_conv = abs(alpha_conv)>0.00018*max(abs(alpha_conv)); 
% Ind_alpha_conv is support recovered from blurred image; For actual recovery without oracle
% info
Ind_alpha_x = abs(alpha_x)>0.0005*max(abs(alpha_x)); 
% Ind_alpha_x is support recovered from original image; For oracle assisted
% recovery 

% Ind_alpha_x = zeros(1,length(alpha)); % Do this if you want to kill support info. from original image
Ind_alpha_conv = zeros(1,length(alpha)); % Do this if you want to kill support info. from blurred image
Ind = ((Ind_alpha_conv>0)|(Ind_alpha_x>0)); % Taking union of both supports

fprintf('Number of non-zeros in x estimated from the blurred image: %.3d\n', sum(Ind_alpha_conv));
fprintf('Number of non-zeros in x estimated from the original image: %.3d\n', sum(Ind_alpha_x));
fprintf('Union of the non-zero support from original and blurred image: %.3d\n', sum(Ind));

figure;
plot(Ind);

%% Compute matrix C; see blind deconvolution paper for notations

j = 1;
N = sum(Ind);
C = sparse(size(alpha,2),N);
for i = 1:size(alpha,2)
    if(Ind(i) == 1)
        C(i,j) = Ind(i);
        m(j) = alpha(i);
        j = j+1;
    end
end
m = m';

%% Define function CC

[c,l] = wavedec2(conv_wx_image,4,'db1');
CC = @(x) waverec2(C*x,l,'db1');
CCT = @(x) (C'*(wavedec2(x,4,'db1'))');

%% Approximated convolved image

x_hat = waverec2(C*m,l,'db1');
fprintf('Origonal image vs Wavelet approximated image: %.3e\n', norm(x-x_hat,'fro')/norm(x,'fro'));
figure;
imagesc(x_hat), title('Approximation of original image from few coeffs'), colormap(gray), colorbar; 

%% Convex Program for deconvolution

[M,H] = blindDeconvolve_implicit_2D(vec(conv_wx),CC,BB,4,CCT,BBT);

[UM,SM,VM] = svd(M,'econ');
[UH,SH,VH] = svd(H,'econ');
[U2,S2,V2] = svd(SM*VM'*VH*SH);

%% Estimates of m and h and recovery errors

mEst=sqrt(S2(1,1))*UM*U2(:,1);
hEst=sqrt(S2(1,1))*UH*V2(:,1);

fprintf('outer product error: %.3e\n', norm(m*h'-mEst*hEst','fro')/norm(m)/norm(h))
fprintf('z error: %.3e\n', norm(m-mEst)/norm(m));
fprintf('h error: %.3e\n', norm(h-hEst)/norm(h));

%% Estimates of x and w

xEst = CC(mEst);
xEst1 = (x(1,1)/xEst(1,1))*(xEst-min(min(xEst)));  % Computing the estimate with an scaling factor
xEst2 = CC((m(1)/xEst(1))*mEst); % Computing the estimate with another scaling factor
wEst = BB(hEst);
figure;
imagesc(xEst1), colormap(gray), colorbar;
figure; 
imagesc(xEst2), colormap(gray), colorbar;
figure;
imagesc(wEst),colormap(gray),colorbar, title('Estimated Kernel');

fprintf('xEst1 error: %.3e\n', norm(x-xEst1,'fro')/norm(x,'fro'));
fprintf('xEst2 error: %.3e\n', norm(x-xEst2,'fro')/norm(x,'fro'));
fprintf('h error: %.3e\n', norm(h-(hEst*h(1)/hEst(1))/norm(h)));

%% Background shading correction for the estimated shapes images

im_Ind = xEst2>0.0042;
for i = 1:256
    for j = 1:256
        if im_Ind(i,j) == 1
            xEst2(i,j) = x(i,j);
        end
    end
end
figure; 
imagesc(xEst2),colormap(gray),colorbar;
