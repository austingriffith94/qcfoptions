% Part A: price = 26.88
% Part B: price = 94.33
% Part C: dt = 0.0100 , price = 7.27
% Part C: dt = 0.0010 , price = 22.73
% Part C: dt = 0.0001 , price = 72.49
% The price of the option for part C depends on the value of dt.
% As dt decreases, the price increases due to more frequent passes through 
% the threshold value.

clc
clear all
clear all

% parameters
S0 = 100;
r = 0.03;
mu = 0.05;
sigma = 0.2;
T = 2;

% paths and intervals
dt = 0.001;
intervals = T/dt;
paths = 10000;

% time and discounts
time_intervals = (0:dt:T);
time_matrix = time_intervals'*ones(1,paths);
discount = exp(-r*time_matrix);

% model movement of underlying
draw = randn(intervals,paths);
dW = sqrt(dt)*(-1 + 2*(draw > .5));
dlogS = (r - 0.5*sigma^2)*dt + sigma*dW;
mid = [log(S0)*ones(1,paths)
    dlogS];
logS = cumsum(mid);
S = exp(logS);

%%
% part A
% exotic option pays maximum of underlying
% over time T - K
K = 100;
maxS = max(S);
payoff_A = max(maxS - K,0);
price_A = payoff_A.*exp(-r*T);
mean_price_A = mean(price_A,2);

fprintf('Part A: price = %.2f\n', ...
        mean_price_A)
    
%%
% part B
% pays average over [0,t], where t is 
% the time when the underlying passes 110
barrier_B = 110;
s_barrier = (S > 110);
temp = time_matrix.*s_barrier;
temp(temp == 0) = T;
first_barrier_time = min(temp);

time_match = ones(intervals+1,paths).*first_barrier_time;
payment_day = time_matrix - time_match;
payment_day(payment_day ~= 0) = NaN;
payment_day = payment_day + 1;
payment_day(payment_day ~= 1) = 0;

N_steps = cumsum(ones(intervals+1,paths));
rolling_avg = cumsum(S)./N_steps;
payoff_B = rolling_avg.*payment_day;
discounted_payoff_B = max(discount.*payoff_B);
mean_price_B = mean(discounted_payoff_B,2);

fprintf('Part B: price = %.2f\n', ...
        mean_price_B)

%%
% part C
% pays one whenever stock crosses 105

% paths and intervals vary with time step
% use loop to iterate through each of the time steps
dt_C = [0.01,0.001,0.0001];

for t = 1:length(dt_C)
    dt = dt_C(t);

    intervals = T/dt;
    paths = 10000000*dt;

    % time and discounts
    time_intervals = (0:dt:T);
    time_matrix = time_intervals'*ones(1,paths);
    discount = exp(-r*time_matrix);
    
    % model movement of underlying
    dW = sqrt(dt)*randn(intervals,paths);
    dlogS = (r - 0.5*sigma^2)*dt + sigma*dW;
    mid = [log(S0)*ones(1,paths)
        dlogS];
    logS = cumsum(mid);
    S = exp(logS);
    
    % calculate price of option from part C
    payment_C = 1;
    barrier_C = 105;
    pass_barrier = S - barrier_C;
    
    % moving product to find when underlying crosses barrier
    sign_change = movprod(pass_barrier,2,1);
    sign_change(1,:) = ones(1,paths);
    sign_change(sign_change == 0) = -1;
    sign_change(sign_change > 0) = 0;
    sign_change(sign_change < 0) = payment_C;

    % cumulative sum of discounted payoffs
    payoff_C = cumsum(sign_change.*discount);
    mean_price_C = mean(payoff_C(end,:),2);
    fprintf('Part C: dt = %.4f , price = %.2f\n', ...
        dt,mean_price_C)
end  