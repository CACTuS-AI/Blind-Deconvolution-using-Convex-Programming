%% Blind Deconvolution using Convex Programming--Stable Recovery
%% (Figure 5(a))
%% Cli Ahmed 
clear all;
%% Path

addpath(fullfile('..','minFunc'));
addpath(fullfile('..','minFunc_2012'));
addpath(fullfile('..','minFunc','compiled'));
addpath(fullfile('..','minFunc','mex'));

%% Parameters 

 L = 2^11;
 N = 500;
 K = 250;
 J = 20; % Number of iterations for each point
 SNR = zeros(7,J);
 MSE = zeros(7,J);
 RMSlog = zeros(J);
for i = 1:7
    for j = 1:J
        idx2 = randperm(L);
        idx2 = idx2(1:K);
        B = speye(L);
        B = B(:,idx2); % Columns random subset of Identity matrix
        % B = B(:,1:K); % First K columns of Identity matrix
        
        scaling = [1 0.5 0.1 0.01 0.001 0.0001 0.00001];
        noise = randn(L,1);
        noise = scaling(i)*noise/norm(noise); 


        m = randn(N,1);
        m= m/norm(m);
        h = randn(K,1);
        h = h/norm(h);
        
        C = randn(L,N)/sqrt(L);
        CC = @(x) C*x;
        CCT = @(x) C'*x;
        BBT = @(x) B'*x;
        BB = @(x) B*x;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        conv_wx = real(ifft(fft((CC(m))).*fft(BB(h))));
        norm(conv_wx)
        conv_wx = conv_wx+noise;
        

        [M,H] = blindDeconvolve_implicit(conv_wx,CC,BB,4,CCT,BBT);

        [UM,SM,VM] = svd(M,'econ');
        [UH,SH,VH] = svd(H,'econ');

        [U2,S2,V2] = svd(SM*VM'*VH*SH);
        mEst=sqrt(S2(1,1))*UM*U2(:,1);
        hEst=sqrt(S2(1,1))*UH*V2(:,1);  
        
        SNR(i,j)  = norm(m*h','fro')^2/norm(noise)^2;
        MSE(i,j)  = norm(m*h'-mEst*hEst','fro')^2/norm(m)^2/norm(h)^2;
        RMSlog(i,j) =  10*log10(norm(m*h'-mEst*hEst','fro')/norm(m)/norm(h));
        fprintf('outer product error: %.3e\n', norm(m*h'-mEst*hEst','fro')/norm(m)/norm(h))
        fprintf('m error: %.3e\n', norm(m-mEst)/norm(m))
        fprintf('h error: %.3e\n', norm(h-hEst)/norm(h))
    end
end
SNR_log = 10*log10(sum(SNR,2)/J);
MSE_log = 10*log10(sum(MSE,2)/J);
       
plot(SNR_log, MSE_log), xlabel('SNR (dB)'), ylabel('Relative  error (dB)');        

