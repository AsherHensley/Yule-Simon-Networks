% Experiment_1: Topology Learning
close all
clear
clc

% Setup Simulation         
N = 500;                % Number of samples to generate
alpha = 1;              % Yule-Simon parameter
seed = 1;               % Random number generator seed
A = [                   % Adjacency Matrix
    0.0, 0.9, 0.9;...
    0.0, 0.0, 0.0;...
    0.0, 0.0, 0.0];

% Synthesize Data Set
[y,xtrue] = synthesis_example(N,seed,A,alpha);

% Run Gibbs Sampler(s)
ngibbs = 2000;
seed = 3;
Chain3 = gibbs_sampler(y,ngibbs,seed,A*nan,alpha);
seed = 10;
Chain10 = gibbs_sampler(y,ngibbs,seed,A*nan,alpha);

% Plot Likelihood
figure,
plot(Chain3.logL),hold on,grid on
plot(Chain10.logL),hold on,grid on
ax = axis;
xlim([-50,ax(2)])
xlabel('MCMC Iteration')
ylabel('Log-Likelihood')
legend('Seed = 3','Seed = 10')

% Plot Adjacency Posterior
twister = [1,4,7,2,5,8,3,6,9];
burn_in = 100;
down_sample = 1;
b = linspace(0,1,200);
figure,
for ii = 1:9
    subplot(3,3,twister(ii))
    [H,b] = hist(Chain3.History.A(ii,burn_in:down_sample:end),b);
    yyaxis left
    plot(b,H/sum(H)), grid on, hold on
    [H,b] = hist(Chain10.History.A(ii,burn_in:down_sample:end),b);
    yyaxis right
    plot(b,H/sum(H)), grid on
    legend('Seed = 3','Seed = 10')
end



