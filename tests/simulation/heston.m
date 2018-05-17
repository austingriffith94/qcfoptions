clc
clear all
close all

% parameters
sigma = 0.30;
r = 0.03; 
S_0 = 100;
T = 2;

% volatility motion parameters
v_0 = sigma^2;
rho = -0.4; % correlation price to volatility
kappa = 25; % speed of adjustment
xi = 0.25^2; % volatility of volatility 

dt = (1/252)/T;
time = (dt:dt:T);

samplePaths = 20000;

%%
% simulating independent Brownians
dW_big = sqrt(dt)*(-1+2*(rand(length(time),2*samplePaths) > 0.5));
dW_S = dW_big(:,1:samplePaths);
dW_V = rho*dW_S + sqrt(1 - rho^2)*(dW_big(:,(samplePaths+1):2*samplePaths));

% volatility simulation
volMotion = nan(length(time)+1,samplePaths);
volMotion(1,:) = v_0; % initializing the volatility

for t = 1:length(time)
    v_t = volMotion(t,:);
    dv_t = kappa*( sigma^2 - v_t )*dt + xi*sqrt(v_t).*dW_V(t,:);
    volMotion(t+1,:) = v_t+dv_t;
end

% use volatility to simulate underlying motion
V = volMotion(1:end-1,:); 
dLog_S = (r - 0.5*V)*dt + sqrt(V).*dW_S;
LogS = cumsum([log(S_0)*ones(1,samplePaths);dLog_S]);
stockMotion = exp(LogS);

%%
% pays average volatility over [0,t], where t is 
% the time when the underlying passes the specified barrier value
timeMatrix = [0,time]'*ones(1,samplePaths);
discount = exp(-r*timeMatrix);

barrier = 110;
sBarrier = (stockMotion > barrier);
temp = timeMatrix.*sBarrier;
temp(temp == 0) = T;
firstBarrierTime = min(temp);

timeMatch = ones(1 + T/dt,samplePaths).*firstBarrierTime;
paymentDay = timeMatrix - timeMatch;
paymentDay(paymentDay ~= 0) = NaN;
paymentDay = paymentDay + 1;
paymentDay(paymentDay ~= 1) = 0;

Nsteps = cumsum(ones(1 + T/dt,samplePaths));
rollingAvg = cumsum(volMotion)./Nsteps; % value set to volatility as payment type
payoff = rollingAvg.*paymentDay;
payoffDiscount = max(discount.*payoff);
priceBarrier = mean(payoffDiscount,2);

fprintf('Barrier Volatility Option Price: $%.3f\n',priceBarrier)