%% Blind Deconvolution using Convex Programming
%% Large scale simulations with different choices of C
%% Ali Ahmed 
clear all;
%% Path

addpath(fullfile('..','minFunc'));
addpath(fullfile('..','minFunc_2012'));
addpath(fullfile('..','minFunc','compiled'));
addpath(fullfile('..','minFunc','mex'));
addpath(fullfile('..','Romberg_noiselet','Measurements'));
addpath(fullfile('..','Romberg_noiselet','Optimization'));
addpath(fullfile('..','Romberg_noiselet','Utils'));

%% Parameters 

L = 2^16;
N = 10000; 
K = 10000;

%% Randomly generate m and h

m = randn(N,1);
m = m/norm(m);
h = randn(K,1);
h = h/norm(h);

%% Coding matrix C
%-------------------------------------------------------------------------%
% Case-1: C is random subset of the columns of Identity matrix
idx1 = randperm(L);
idx1 = idx1(1:N);
C = speye(L);
C = C(:,idx1); % sparse
CC = @(x)C*x;
CCT = @(x)C'*x;
%-------------------------------------------------------------------------%
% Case-2: C is a random Gaussian matrix
% C = randn(L,N)/sqrt(L);
% CC = @(x)C*x;
% CCT = @(x)C'*x;
%-------------------------------------------------------------------------%
% Case-3: C is a noislet matrix 
% A = speye(L);
% A = A(:,1:N);
% q       = randperm(L)';    % makes column vector of random integers 1:N
% OM2     = q(1:L);          % vector of random subset of integers 1:N
% CC     = @(x) A_noiselet (A*x, OM2);
% CCT    = @(x) A'*At_noiselet(x, OM2, L);
%-------------------------------------------------------------------------%
% Case 4: C is a random subset of the columns of wavelet matrix
% idx1 = randperm(L);
% idx1 = idx1(1:N);
% C = speye(L);
% C = C(:,idx1); % sparse
% [alpha,l] = wavedec(C*m,3,'haar');
% CC = @(x) wavedec(C*x,3,'haar');
% CCT = @(x) (C'*waverec(x,l,'haar'));
%-------------------------------------------------------------------------%
%% Matrix B
idx2 = randperm(L);
idx2 = idx2(1:K);
B = speye(L);
B = B(:,idx2); % sparse
% B = B(:,1:K); % short
BB = @(x)B*x;
BBT = @(x) B'*x;

%% Convolve x and w
conv_wx = real(ifft(fft(CC(m)).*fft(BB(h))));

%% Deconvolve
[M,H] = blindDeconvolve_implicit(conv_wx,CC,BB,4,CCT,BBT);

[UM,SM,VM] = svd(M,'econ');
[UH,SH,VH] = svd(H,'econ');

[U2,S2,V2] = svd(SM*VM'*VH*SH);
mEst=sqrt(S2(1,1))*UM*U2(:,1);
hEst=sqrt(S2(1,1))*UH*V2(:,1);
error = norm(m*h'-mEst*hEst','fro')/norm(m)/norm(h);

fprintf('outer product error: %.3e\n', norm(m*h'-mEst*hEst','fro')/norm(m)/norm(h))
fprintf('z error: %.3e\n', norm(m-mEst)/norm(m))
fprintf('h error: %.3e\n', norm(h-hEst)/norm(h))

