'''
Austin Griffith
bsoptions.py
'''

import numpy as np
from scipy.stats import norm

# options
def Euro(s,k,r,T,vol,q):
    '''
    Calculate the Black Scholes value of a European Call / Put option
    Payoffs are of the form :
    C = max(S - K, 0)
    P = max(K - S, 0)

    Parameters
    ----------
    s : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying s
        Spot value of underlying asset at current time, t
    k : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying k
        Strike value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying r
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying T
        Time till expiration for option
    vol : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying vol
        Volatility of underlying, implied constant till expiration in Black
        Scholes model
    q : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying q
        Continuous dividend payout, as a percentage

    Notes
    -----
    All parameters can be individual values.
    Only one of these parameters can be a numpy.array, otherwise there will be
    a dimension mismatch.

    Returns
    -------
    [call,put] : list of pair of float or numpy.array values
        European call and put values, type depends on input value.
        If all input values are individual numbers, then output will be float.
        If one input value is numpy.array, then output will be numpy.array.

    '''
    d1 = ((np.log(s/k) + (r - q + 0.5*vol*vol)*T)) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)

    option = np.exp(-q*T)*s
    strike = np.exp(-r*T)*k

    put = strike*norm.cdf(-d2) - option*norm.cdf(-d1)
    call = option*norm.cdf(d1) - strike*norm.cdf(d2)
    return([call,put])

def AsianGeometric(s,k,r,T,vol,q):
    '''
    Calculate the Black Scholes value of Geometric Average Asian Call / Put
    option with a fixed strike
    Payoffs are of the form :
    C = max(AVG_geo - K, 0)
    P = max(K - AVG_geo, 0)

    Parameters
    ----------
    s : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying s
        Spot value of underlying asset at current time, t
    k : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying k
        Strike value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying r
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying T
        Time till expiration for option
    vol : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying vol
        Volatility of underlying, implied constant till expiration in Black
        Scholes model
    q : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying q
        Continuous dividend payout, as a percentage

    Notes
    -----
    All parameters can be individual values.
    Only one of these parameters can be a numpy.array, otherwise there will be
    a dimension mismatch.

    Returns
    -------
    [call,put] : list of pair of float or numpy.array values
        Asian call and put values, type depends on input value.
        If all input values are individual numbers, then output will be float.
        If one input value is numpy.array, then output will be numpy.array.

    '''
    a = 0.5*(r - q - vol*vol/6)
    volG = vol/np.sqrt(3)

    d1 = (np.log(s/k) + (a + 0.5*volG*volG)*T) / (volG*np.sqrt(T))
    d2 = d1 - volG*np.sqrt(T)

    option = s*np.exp((a - r)*T)
    strike = k*np.exp(-r*T)

    put = strike*norm.cdf(-d2) - option*norm.cdf(-d1)
    call = option*norm.cdf(d1) - strike*norm.cdf(d2)
    return([call,put])

def AsianArithmetic(s,k,r,T,vol,q):
    '''
    Calculate the Black Scholes value of Arithmetic Average Asian Call / Put
    option with a fixed strike
    Payoffs are of the form :
    C = max(AVG_arithmetic - K, 0)
    P = max(K - AVG_arithmetic, 0)

    Parameters
    ----------
    s : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying s
        Spot value of underlying asset at current time, t
    k : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying k
        Strike value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying r
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying T
        Time till expiration for option
    vol : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying vol
        Volatility of underlying, implied constant till expiration in Black
        Scholes model
    q : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying q
        Continuous dividend payout, as a percentage

    Notes
    -----
    All parameters can be individual values.
    Only one of these parameters can be a numpy.array, otherwise there will be
    a dimension mismatch.

    * want r > q, else the natural logarithm has a chance of breaking

    Returns
    -------
    [call,put] : list of pair of float or numpy.array values
        Asian call and put values, type depends on input value.
        If all input values are individual numbers, then output will be float.
        If one input value is numpy.array, then output will be numpy.array.

    '''
    m1 = s*(np.exp((r - q)*T) - 1) / ((r - q)*T)
    m2l = 2*s*s*np.exp((2*r - 2*q + vol*vol)*T) / ((r - q +vol*vol)*T*T*(2*r - 2*q + vol*vol))
    m2r = (2*s*s / ((r-q)*T*T))*((1/(2*(r-q) + vol*vol)) -
        np.exp((r-q)*T)/(r - q - vol*vol))
    m2 = m2l + m2r

    volA = np.sqrt(np.log(m2/(m1*m1)) / T)

    d1 = (np.log(m1/k) + 0.5*volA*volA*T) / (volA*np.sqrt(T))
    d2 = d1 - volA*np.sqrt(T)

    call = np.exp(-r*T)*(m1*norm.cdf(d1) - k*norm.cdf(d2))
    put = np.exp(-r*T)*(k*norm.cdf(-d2) - m1*norm.cdf(-d1))
    return([call,put])

def Power(s,k,r,T,vol,q,n):
    '''
    Calculate the Black Scholes value of a traditional Power Call / Put option
    with a fixed strike
    Payoffs are of the form :
    C = max(S**n - K, 0)
    P = max(K - S**n, 0)

    Parameters
    ----------
    s : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying s
        Spot value of underlying asset at current time, t
    k : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying k
        Strike value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying r
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying T
        Time till expiration for option
    vol : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying vol
        Volatility of underlying, implied constant till expiration in Black
        Scholes model
    q : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying q
        Continuous dividend payout, as a percentage
    n : number of any type (int, float8, float64 etc.)
        Power to which the underlying spot is raised at payoff

    Notes
    -----
    All parameters can be individual values.
    Only one of these parameters can be a numpy.array (not including 'n'),
    otherwise there will be a dimension mismatch.

    Returns
    -------
    [call,put] : list of pair of float or numpy.array values
        Power call and put values, type depends on input value.
        If all input values are individual numbers, then output will be float.
        If one input value, other than 'n', is numpy.array, then output
        will be numpy.array.

    '''
    d1 = (np.log(s/np.power(k,1/n)) +
        (r - q + vol*vol*(n - 0.5))*T) / (vol*np.sqrt(T))
    d2 = d1 - n*vol*np.sqrt(T)

    option = np.exp(T*(n-1)*(r + 0.5*n*vol*vol))*np.power(s,n)
    strike = k*np.exp(-r*T)

    put = strike*norm.cdf(-d2) - option*norm.cdf(-d1)
    call = option*norm.cdf(d1) - strike*norm.cdf(d2)
    return([call,put])

def PowerStrike(s,k,r,T,vol,q,n):
    '''
    Calculate the Black Scholes value of Power Call / Put option with a
    fixed strike to the power n
    Payoffs are of the form :
    C = max(S**n - K**n, 0)
    P = max(K**n - S**n, 0)

    Parameters
    ----------
    s : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying s
        Spot value of underlying asset at current time, t
    k : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying k
        Strike value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying r
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying T
        Time till expiration for option
    vol : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying vol
        Volatility of underlying, implied constant till expiration in Black
        Scholes model
    q : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying q
        Continuous dividend payout, as a percentage
    n : number of any type (int, float8, float64 etc.)
        Power to which the underlying spot is raised at payoff

    Notes
    -----
    All parameters can be individual values.
    Only one of these parameters can be a numpy.array (not including 'n'),
    otherwise there will be a dimension mismatch.

    Returns
    -------
    [call,put] : list of pair of float or numpy.array values
        Power call and put values, type depends on input value.
        If all input values are individual numbers, then output will be float.
        If one input value, other than 'n', is numpy.array, then output
        will be numpy.array.

    '''
    d1 = (np.log(s/np.power(k,1/n)) +
        (r - q + vol*vol*(n - 0.5))*T) / (vol*np.sqrt(T))
    d2 = d1 - n*vol*np.sqrt(T)

    option = np.exp(T*(n-1)*(r + 0.5*n*vol*vol))*np.power(s,n)
    strike = np.power(k,n)*np.exp(-r*T)

    put = strike*norm.cdf(-d2) - option*norm.cdf(-d1)
    call = option*norm.cdf(d1) - strike*norm.cdf(d2)
    return([call,put])

def Margrabe(s,s2,T,vol,vol2,q,q2,corr):
    '''
    Calculate the Black Scholes value of the Margrabe Option
    Payoff is of the form :
    O = max(S_1 - S_2, 0)

    Parameters
    ----------
    s1 and s2 : number of any type (int, float8, float64 etc.), numpy array of
        any type should the user wish to have a list of values output with
        varying s
        Spot value of underlying assets 1 and 2 at current time, t
    T : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying T
        Time till expiration for option
    vol1 and vol2 : number of any type (int, float8, float64 etc.), numpy array
        of any type should the user wish to have a list of values output with
        varying vol
        Volatility of underlying for assets 1 and 2, implied constant till
        expiration in Black Scholes model
    q1 and q2 : number of any type (int, float8, float64 etc.), numpy array of
        any type should the user wish to have a list of values output with
        varying q
        Continuous dividend payout for assets 1 and 2, as a percentage
    corr : number of any type (int, float8, float64 etc.), numpy array of any
        type should the user wish to have a list of values output with varying
        corr
        Correlation between the motion of the underlying (relationship between
        the Weiner process of asset 1 and 2)

    Notes
    -----
    All parameters can be individual values.
    At most, only one pair of these parameters can be a numpy.array,
    otherwise there will be a dimension mismatch.

    Returns
    -------
    price : float or numpy.array value
        Margrabe price, type depends on input value.
        If all input values are individual numbers, then output will be float.
        If one pair of input values are a numpy.array,
        then output will be numpy.array.

    '''
    volMix = np.sqrt(vol*vol + vol2*vol2 - vol*vol2*corr)
    d1 = (np.log(s/s2) + (q2 - q + 0.5*(volMix**2))*T) / (volMix*np.sqrt(T))
    d2 = d1 - volMix*np.sqrt(T)

    option = np.exp(-q*T)*s*norm.cdf(d1)
    option2 = np.exp(-q2*T)*s2*norm.cdf(d2)

    price = option - option2
    return(price)

def Lookback(s,M,r,T,vol,q):
    '''
    Calculate the Black Scholes value of floating strike
    Lookback Call / Put option
    Payoffs are of the form :
    C = S_T - min(m,m_T)
    P = max(M,M_T) - S_T
    where 'm' is the current minimum, or starting strike at initiation, and
    'm_T' is the minimum over the remaining life of the option
    similarly, 'M' is the current maximum, or starting strike at initiation, and
    'M_T' is the maximum over the remaining life of the option

    Parameters
    ----------
    s : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying s
        Spot value of underlying asset at current time, t
    M : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying k
        Strike value of option, determined by minimum value of underlying
        over the life of the option
    r : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying r
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying T
        Time till expiration for option
    vol : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying vol
        Volatility of underlying, implied constant till expiration in Black
        Scholes model
    q : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying q
        Continuous dividend payout, as a percentage

    Notes
    -----
    All parameters can be individual values.
    Only one of these parameters can be a numpy.array, otherwise there will be
    a dimension mismatch.

    Returns
    -------
    [call,put] : list of pair of float or numpy.array values
        Lookback call and put values, type depends on input value.
        If all input values are individual numbers, then output will be float.
        If one input value is numpy.array, then output will be numpy.array.

    '''
    B = 2*(r - q) / (vol*vol)
    x = (np.log(s/M) + (r - q - 0.5*vol*vol)*T) / (vol*np.sqrt(T))
    y = (-np.log(s/M) - (r - q + 0.5*vol*vol)*T) / (vol*np.sqrt(T))

    option = s*np.exp(-q*T)
    minimum = M*np.exp(-r*T)
    left = np.exp(-r*T)*np.power(s/M,-B)
    right = np.exp(-q*T)

    call = (option*norm.cdf(x + vol*np.sqrt(T)) -
        minimum*norm.cdf(x) +
        (s/B)*(left*norm.cdf(y + B*vol*np.sqrt(T)) -
            right*norm.cdf(y)))
    put = (-option*norm.cdf(-x - vol*np.sqrt(T)) +
        minimum*norm.cdf(-x) -
        (s/B)*(left*norm.cdf(-y - B*vol*np.sqrt(T)) -
            right*norm.cdf(-y)))
    return([call,put])

# greeks
def EuroDelta(s,k,r,T,vol,q):
    '''
    Calculate the Black Scholes delta value of a European Call / Put option,
    measures sensitivity of option value with respect to change in underlying
    asset price, calculated from derivative dV/dS
    Deltas are of the form :
    C = e**-qT N(d1)
    P = -e**-qT N(-d1)

    Parameters
    ----------
    s : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying s
        Spot value of underlying asset at current time, t
    k : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying k
        Strike value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying r
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying T
        Time till expiration for option
    vol : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying vol
        Volatility of underlying, implied constant till expiration in Black
        Scholes model
    q : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying q
        Continuous dividend payout, as a percentage

    Notes
    -----
    All parameters can be individual values.
    Only one of these parameters can be a numpy.array, otherwise there will be
    a dimension mismatch.

    Returns
    -------
    [call,put] : list of pair of float or numpy.array values
        European call and put delta values, type depends on input value.
        If all input values are individual numbers, then output will be float.
        If one input value is numpy.array, then output will be numpy.array.

    '''
    d1 = ((np.log(s/k) + (r - q + 0.5*vol*vol)*T)) / (vol*np.sqrt(T))

    call = np.exp(-q*T)*norm.cdf(d1)
    put = -np.exp(-q*T)*norm.cdf(-d1)
    return([call,put])

def EuroGamma(s,k,r,T,vol,q):
    '''
    Calculate the Black Scholes gamma value of a European Call / Put option,
    measures sensitivity of option delta with respect to change in underlying
    asset price, calculated from derivative d2C/dS2
    Gamma is of the form :
    G = e**(-qT) N'(d1) / S vol sqrt(T)

    Parameters
    ----------
    s : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying s
        Spot value of underlying asset at current time, t
    k : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying k
        Strike value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying r
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying T
        Time till expiration for option
    vol : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying vol
        Volatility of underlying, implied constant till expiration in Black
        Scholes model
    q : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying q
        Continuous dividend payout, as a percentage

    Notes
    -----
    All parameters can be individual values.
    Only one of these parameters can be a numpy.array, otherwise there will be
    a dimension mismatch.

    Returns
    -------
    gamma : float or numpy.array value
        European gamma value, type depends on input value.
        If all input values are individual numbers, then output will be float.
        If one input value is numpy.array, then output will be numpy.array.

    '''
    d1 = ((np.log(s/k) + (r - q + 0.5*vol*vol)*T)) / (vol*np.sqrt(T))
    gamma = np.exp(-q*T)*norm.pdf(d1) / (s*vol*np.sqrt(T))
    return(gamma)

def EuroVega(s,k,r,T,vol,q):
    '''
    Calculate the Black Scholes vega value of a European Call / Put option,
    measures sensitivity of option value with respect to change in underlying
    asset volatility, calculated from derivative dV/dσ
    Vega is of the form :
    V = S e**(-qT) N'(d1) sqrt(T)

    Parameters
    ----------
    s : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying s
        Spot value of underlying asset at current time, t
    k : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying k
        Strike value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying r
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying T
        Time till expiration for option
    vol : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying vol
        Volatility of underlying, implied constant till expiration in Black
        Scholes model
    q : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying q
        Continuous dividend payout, as a percentage

    Notes
    -----
    All parameters can be individual values.
    Only one of these parameters can be a numpy.array, otherwise there will be
    a dimension mismatch.

    Returns
    -------
    vega : float or numpy.array value
        European vega value, type depends on input value.
        If all input values are individual numbers, then output will be float.
        If one input value is numpy.array, then output will be numpy.array.

    '''
    d1 = ((np.log(s/k) + (r - q + 0.5*vol*vol)*T)) / (vol*np.sqrt(T))
    vega = s*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T)
    return(vega)

def EuroTheta(s,k,r,T,vol,q):
    '''
    Calculate the Black Scholes theta value of a European Call / Put option,
    measures sensitivity of option value with respect to change in time,
    calculated from derivative dV/dT

    Parameters
    ----------
    s : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying s
        Spot value of underlying asset at current time, t
    k : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying k
        Strike value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying r
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying T
        Time till expiration for option
    vol : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying vol
        Volatility of underlying, implied constant till expiration in Black
        Scholes model
    q : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying q
        Continuous dividend payout, as a percentage

    Notes
    -----
    All parameters can be individual values.
    Only one of these parameters can be a numpy.array, otherwise there will be
    a dimension mismatch.

    Returns
    -------
    [call,put] : list of pair of float or numpy.array values
        European call and put theta values, type depends on input value.
        If all input values are individual numbers, then output will be float.
        If one input value is numpy.array, then output will be numpy.array.

    '''
    d1 = ((np.log(s/k) + (r - q + 0.5*vol*vol)*T)) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)

    deriv = -0.5*np.exp(-q*T)*norm.pdf(d1)*s*vol / np.sqrt(T)
    strike = r*k*np.exp(-r*T)
    option = q*s*np.exp(-q*T)

    call = deriv - strike*norm.cdf(d2) + option*norm.cdf(d1)
    put = deriv + strike*norm.cdf(-d2) - option*norm.cdf(-d1)
    return([call,put])

def EuroRho(s,k,r,T,vol,q):
    '''
    Calculate the Black Scholes rho value of a European Call / Put option,
    measures sensitivity of option value with respect to change in interest
    rate over the life of the option, calculated from derivative dV/dr

    Rhos are of the form :
    C = e**-rT K T N(d2)
    P = -e**-rT K T N(-d2)

    Parameters
    ----------
    s : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying s
        Spot value of underlying asset at current time, t
    k : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying k
        Strike value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying r
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying T
        Time till expiration for option
    vol : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying vol
        Volatility of underlying, implied constant till expiration in Black
        Scholes model
    q : number of any type (int, float8, float64 etc.), numpy array of any type
        should the user wish to have a list of values output with varying q
        Continuous dividend payout, as a percentage

    Notes
    -----
    All parameters can be individual values.
    Only one of these parameters can be a numpy.array, otherwise there will be
    a dimension mismatch.

    Returns
    -------
    [call,put] : list of pair of float or numpy.array values
        European call and put rho values, type depends on input value.
        If all input values are individual numbers, then output will be float.
        If one input value is numpy.array, then output will be numpy.array.

    '''
    d1 = ((np.log(s/k) + (r - q + 0.5*vol*vol)*T)) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)

    strike = k*T*np.exp(-r*T)

    call = strike*norm.cdf(d2)
    put = -strike*norm.cdf(-d2)
    return([call,put])
