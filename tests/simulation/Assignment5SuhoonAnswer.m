clc
clear all
close all

%% parameters
r = 0.03;
sigma = 0.2;
mu = 0.05;
T = 2;
S_0 = 100;
dt = 1/100;
number_of_sample_paths = 1000;
number_of_small_intervals = T/dt;

%% simulate sample path of stock price

uniform_draw = rand(number_of_small_intervals,number_of_sample_paths);
binomial_one_or_minus_one = -1 + 2*(uniform_draw > .5);

dW = sqrt(dt) * binomial_one_or_minus_one;
dlogS = (r - 0.5*sigma^2)*dt + sigma*dW;
temp = [log(S_0)*ones(1,number_of_sample_paths)
    dlogS];
logS = cumsum(temp);
S = exp(logS);

%% question a

maximum_price = max(S);
payoff = max(maximum_price-100,0);
discounted_payoff = payoff.*exp(-r*T);
answer_a = mean(discounted_payoff);

%% question b

time = (0:dt:T);
time_matrix = time'*ones(1,number_of_sample_paths);
whether_S_is_below_barrier = (S < 110);
whether_S_never_hits_barrier_so_far = cumprod( whether_S_is_below_barrier );

temp = time_matrix.*whether_S_never_hits_barrier_so_far;
payment_time = max( temp );
average_price_at_the_payment = sum( whether_S_never_hits_barrier_so_far.*S ) ./ sum( whether_S_never_hits_barrier_so_far ); % this is the amount of payoff
discounted_payoff = exp(-r*payment_time).*average_price_at_the_payment;

answer_b = mean( discounted_payoff );


%% question c

whether_S_is_below = (S < 105);

temp = whether_S_is_below(1:end-1,:) - whether_S_is_below(2:end,:);
payment_matrix = ( abs(temp) > 0 ); 

time=(dt:dt:T);
time_matrix = time'*ones(1,number_of_sample_paths);

discounted_payment_matrix = exp(-r*time_matrix).*payment_matrix;
discounted_payment_across_sample_paths = sum( discounted_payment_matrix );

answer_c = mean( discounted_payment_across_sample_paths );


