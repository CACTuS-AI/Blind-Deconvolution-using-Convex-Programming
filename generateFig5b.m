%% Test Blind Deconvolution using Convex Programming--Stable Recovery 
%% Figure 5(b)
%% Ali Ahmed 
clear all;
%% Path

addpath(fullfile('..','minFunc'));
addpath(fullfile('..','minFunc_2012'));
addpath(fullfile('..','minFunc','compiled'));
addpath(fullfile('..','minFunc','mex'));

 
 N = 500;
 K = 250;
 Jt = 1;
 SNR = zeros(5,Jt);
 RMS = zeros(5,Jt);
 Oversampling = zeros(5,1);
 RMSlog = zeros(5,Jt);
for i = 1:10
    for j = 1:Jt
        L = 1000*i;
        
        idx2 = randperm(L);
        idx2 = idx2(1:K);
        B = speye(L);
        B = B(:,idx2);


        noise = randn(L,1);
        noise = 0.1*noise/norm(noise);


        m = randn(N,1);
        m= m/norm(m);
        h = randn(K,1);
        h = h/norm(h);
        
        C = randn(L,N)/sqrt(L);
        CC = @(x) C*x;
        CCT = @(x) C'*x;

        
        BBT = @(x) B'*x;
        BB = @(x) B*x;
        conv_wx = real(ifft(fft((CC(m))).*fft(BB(h))))+noise; 

        [M,H] = blindDeconvolve_implicit(conv_wx,CC,BB,4,CCT,BBT);

        [UM,SM,VM] = svd(M,'econ');
        [UH,SH,VH] = svd(H,'econ');

        [U2,S2,V2] = svd(SM*VM'*VH*SH);
        mEst=sqrt(S2(1,1))*UM*U2(:,1);
        hEst=sqrt(S2(1,1))*UH*V2(:,1);  
        Oversampling(i) = L/(N+K);
        SNR(i,j)  = norm(m*h','fro')^2/norm(noise)^2;
        RMS(i,j)  = norm(m*h'-mEst*hEst','fro')/norm(m)/norm(h);
        RMSlog(i,j) =  10*log10(norm(m*h'-mEst*hEst','fro')/norm(m)/norm(h));
        fprintf('outer product error: %.3e\n', norm(m*h'-mEst*hEst','fro')/norm(m)/norm(h))
        fprintf('m error: %.3e\n', norm(m-mEst)/norm(m))
        fprintf('h error: %.3e\n', norm(h-hEst)/norm(h))
    end
end
SNR2 = 10*log10(sum(SNR,2)/Jt);
RMS2 = (sum(RMS,2)/Jt);
    
plot(RMS2,Oversampling), xlabel('Oversampling: L/(N+k)'), ylabel('Relative error');        
