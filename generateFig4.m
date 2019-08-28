%% Blind Deconvolution using Convex Programming
%% Phase transitions--Figure 3
%% Ali Ahmed

%% Path 

addpath(fullfile('..','minFunc'));
addpath(fullfile('..','minFunc_2012'));
addpath(fullfile('..','minFunc','compiled'));
addpath(fullfile('..','minFunc','mex'));

%% 
success_freq = zeros(40,40);
filename = sprintf('Phase_transitions');
L = 2^15;
for i = 1:40
    for j = 1:40
        N = 25*i
        K = 25*j
        for k = 1:1          
%% Matrices B and C  
            C = randn(L,N)/sqrt(L);
            idx2 = randperm(L);
            idx2 = idx2(1:K);
            B = speye(L);
            B = B(:,idx2); % Sparse
%           S2 = S2(:,1:n2); % Short

            %% Random vectors
            
            m = randn(N,1);
            m = m/norm(m);
            h = randn(K,1);
            h = h/norm(h);
            
            %% Convolution
            conv_wx = real(ifft(fft(C*m).*fft(B*h)));
            [M,H] = blindDeconvolve(conv_wx,C,B,4);

            [UM,SM,VM] = svd(M,'econ');
            [UH,SH,VH] = svd(H,'econ');

            [U2,S2,V2] = svd(SM*VM'*VH*SH);
            mEst=sqrt(S2(1,1))*UM*U2(:,1);
            hEst=sqrt(S2(1,1))*UH*V2(:,1);

            error = norm(m*h'-mEst*hEst','fro')/norm(m)/norm(h);
            fprintf('outer product error: %.3e\n', norm(m*h'-mEst*hEst','fro')/norm(m)/norm(h))
            fprintf('z error: %.3e\n', norm(m-mEst)/norm(m))
            fprintf('h error: %.3e\n', norm(h-hEst)/norm(h))
           if(error<2e-2)
                    success_freq(i,j) = success_freq(i,j)+1;
           end
       end
    end
end

imagesc(flipud(success_freq)/100), colormap(gray), colorbar;
save(filename);