% ----------------------------------
% Synthesize Example Data Set 
% ----------------------------------
function [y,x] = synthesis_example(N,SEED,A,alpha)

% Set Random Number Generator
rng(SEED);
    
% Create Latent State Variables
d = size(A,1);
x = zeros(N,d);
for kk = 1:d
    x(:,kk) = ysp(N,alpha);
end

% Create Observations
y = interactions(x,A);

% Plot Results
figure,
subplot(121)
imagesc(~x)
set(gca,'xtick',1:3,'xticklabel',{'A','B','C'})
title('x_t')
colormap bone
ylabel('Sample')
xlabel('Node')
subplot(122)
imagesc(~y)
set(gca,'xtick',1:3,'xticklabel',{'A','B','C'})
colormap bone
title('y_t')
ylabel('Sample')
xlabel('Node')

% ----------------------------------
% Simulate Yule-Simon Process
% ----------------------------------
function x = ysp(N,alpha)
n = 1;
x = zeros(N,1);
for kk = 1:N
    p = alpha/(n+alpha);
    x(kk) = double(rand < p);
    n = (n+1)^(1 - x(kk));
end

% ----------------------------------
% Simulate Network Interactions
% ----------------------------------
function y = interactions(x,A)
y = x * 0;
[N,d] = size(x);
for t = 1:N
    for kk = 1:d
        temp = (1 - A(:,kk)).^(x(t,:)');
        temp(kk) = [];
        beta = (1 - prod(temp))^(1 - x(t,kk));
        y(t,kk) = double(rand < beta);
    end
end





