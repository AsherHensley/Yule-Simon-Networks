% ----------------------------------
% Sample Yule-Simon Network Posterior
% ----------------------------------
function Chain = gibbs_sampler(y,ngibbs,seed,A,alpha)

% Setup Markov Chain
Chain.seed = seed;
Chain.ngibbs = ngibbs;
Chain.logL = zeros(Chain.ngibbs,1);
Chain.tsamp = find(sum(y,2)>1);
Chain.x = y;
Chain.A = A;
Chain.alpha = alpha;
Chain.theta = nan;
Chain.phi = nan;
Chain.History.A = zeros(length(A(:)),ngibbs);

% Init Adjacency Matrix
do_sample_adjacency = false;
if any(isnan(Chain.A(:)))
    Chain.A = rand(size(Chain.A));
    for ii = 1:size(Chain.A,1)
        Chain.A(ii,ii) = 0;
    end
    do_sample_adjacency = true;
end

% Run Gibbs Sampler
hw = waitbar(0,'Simulating Markov Chain...');
rng(Chain.seed);
for jj = 1:Chain.ngibbs
    
    % Yule-Simon Sampler
    Chain.x = sample_yulesimon_process(y,Chain);

    % Adjacency Matrix Sampler
    if do_sample_adjacency==true
        [Chain.A,Chain.theta,Chain.phi] = sample_adjacency_matrix(Chain,y,true);
    end
    
    % Update Model Likelihood
    Chain.logL(jj) = model_likelihood(y,Chain);
    
    % Update History
    Chain.History.A(:,jj) = Chain.A(:);
    
    % Update Waitbar
    if mod(jj,10)==0
        waitbar(jj/Chain.ngibbs,hw,'Simulating Markov Chain...')
    end
    
end
delete(hw)

% ----------------------------------
% Sample Yule-Simon Process
% ----------------------------------
function x = sample_yulesimon_process(y,Chain)

% Setup
[~,d] = size(y);

% Loop Over Data Channels
for kk = 1:d
    
    % Loop Over Time-Stamps
    for t = 1:length(Chain.tsamp)
        
        % Get Current Time Stamp
        tj = Chain.tsamp(t);
        
        % Get State Counter
        state = get_states(Chain.x(:,kk));
        j = state(tj);
        
        % Skip Condition
        if sum(Chain.x(tj,:))==1 && Chain.x(tj,kk)==1
            continue
        end
        
        % Get Boundaries
        left_boundary = find(state==j,1,'first')-1;
        if Chain.x(tj,kk)==0
            right_boundary = find(state==j,1,'last');
        else
            right_boundary = find(state==j+1,1,'last');
        end
        
        % Compute Regime Sizes
        left_regime = tj - left_boundary;
        right_regime = right_boundary - tj;
        
        % Priors
        prior0 = Chain.alpha * beta(left_regime + right_regime, Chain.alpha + 1);
        prior1 = Chain.alpha^2 * beta(left_regime, Chain.alpha + 1) * beta(right_regime, Chain.alpha + 1);
        
        % Likelihoods
        x0 = Chain.x(tj,:);
        x0(kk) = 0;
        likelihood0 = data_likelihood(y(tj,:),x0,Chain.A);
        x1 = Chain.x(tj,:);
        x1(kk) = 1;
        likelihood1 = data_likelihood(y(tj,:),x1,Chain.A);
        
        % Posterior
        posterior = [likelihood0 * prior0, likelihood1 * prior1];
        posterior = posterior/sum(posterior);
        
        % Draw Sample
        Chain.x(tj,kk) = double(rand < posterior(2));
        
    end
end

x = Chain.x;

% ----------------------------------
% Sample Adjacency Matrix
% ----------------------------------
function [A,theta,phi] = sample_adjacency_matrix(Chain,y,do_exclusion)

% Setup
[~,d] = size(y);
A = zeros(d);
N = length(Chain.tsamp);
p = ones(N,d);

% Sample Interaction Indicators
if do_exclusion==true
    event = Chain.x(Chain.tsamp,:)==0 & y(Chain.tsamp,:)==1;
    p = zeros(N,d);
    for kk = 1:N
        interact = find(event(kk,:)==1);
        for ii = 1:length(interact)
            PDF = Chain.A(:,interact(ii)) .* Chain.x(Chain.tsamp(kk),:)';
            ctrl = false(size(PDF));
            while all(ctrl == false)
                ctrl = double(rand(size(PDF)) < PDF);
            end
            p(kk,:) = ctrl;
        end
        
    end
end

% Sample Posterior
theta = zeros(d,d);
phi = zeros(d,d);
for ii = 1:d
    for jj = 1:d
        if ii==jj
            continue
        end
        xi = Chain.x(Chain.tsamp,ii) .* p(:,ii);
        yj = y(Chain.tsamp,jj);
        theta(ii,jj) = 1 + (xi' * yj);
        phi(ii,jj) = 1 + N - (xi' * yj);
        A(ii,jj) = betarnd(theta(ii,jj),phi(ii,jj));
    end
end

% ----------------------------------
% Get Numbered Regime Array
% ----------------------------------
function state = get_states(x)

state = cumsum([1;x]);
state(end) = [];

% ----------------------------------
% Get Data Likelihood
% ----------------------------------
function L = data_likelihood(y,x,A)

% Setup
d = size(A,1);
beta = zeros(1,d);

% Compute betas
for kk = 1:d
    beta(kk) = compute_beta(x,kk,A);
end

% Compute Likelihood
L = prod(beta.^y .* (1-beta).^(1-y));

% ----------------------------------
% Compute Beta Probability
% ----------------------------------
function beta = compute_beta(x,k,A)
temp = (1 - A(:,k)).^x(:);
temp(k) = [];
beta = (1 - prod(temp))^(1 - x(k));
beta = min(max(beta,1e-6),1-1e-6);

% ----------------------------------
% Compute Model Log-Likelihood
% ----------------------------------
function logL = model_likelihood(y,Chain)

% Data Likelihood
[N,d] = size(y);
logL1 = zeros(N,1);
for kk = 1:N
    logL1(kk) = log(data_likelihood(y(kk,:),Chain.x(kk,:),Chain.A));
end

% State Variable Likelihood
logL2 = zeros(1,d);
for kk = 1:d
    state = get_states(Chain.x(:,kk));
    regime = hist(state,unique(state));
    logL2(kk) = sum(log(Chain.alpha) + log(beta(regime,Chain.alpha+1)));
end

% Adjacency Matrix Likelihood
logL3 = zeros(d,d);
if ~isnan(Chain.theta)
    for ii = 1:d
        for jj = 1:d
            if ii==jj
                continue;
            end
            logL3(ii,jj) = -log(beta(Chain.theta(ii,jj),Chain.phi(ii,jj)));
            logL3(ii,jj) = logL3(ii,jj) + (Chain.theta(ii,jj)-1) * log(Chain.A(ii,jj));
            logL3(ii,jj) = logL3(ii,jj)+ (Chain.phi(ii,jj)-1) * log(1-Chain.A(ii,jj));
        end
    end
end

% Complete-Data Log Likelihood
logL = sum(logL1) + sum(logL2) + sum(logL3(:));


