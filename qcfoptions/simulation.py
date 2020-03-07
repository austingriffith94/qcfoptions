'''
Austin Griffith
simulation.py
'''

import numpy as np
import scipy.stats as sctats
import time

# simulation payoffs
def EuroSim(S,k,r,T):
    '''
    Use simulated underlying to determine the price of a European
    Call / Put option
    Payoffs are of the form :
    C = max(S - K, 0)
    P = max(K - S, 0)

    Parameters
    ----------
    S : numpy.array
        Simulated stock price, want to be of the form such that the first row
        is the initial stock price, with subsequent rows representing an
        additional time step increase, and each column is a simulated path of
        the asset
    k : number of any type (int, float8, float64 etc.)
        Strike value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.)
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.)
        Time till expiration for option

    Returns
    -------
    [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
        floats, second of one-dimensional numpy.array's
        First list is the call and put price, determined by the average
        of the simulated stock payoffs
        Second list is the call and put simulated paths payoffs at expiration,
        NOT discounted

    Notes
    -----
    The accuracy of pricing is dependent on the number of time steps and
    simulated paths chosen for the underlying stochastic motion

    '''
    callMotion = (S[-1] - k).clip(0)
    putMotion = (k - S[-1]).clip(0)

    call = np.exp(-r*T)*np.average(callMotion)
    put = np.exp(-r*T)*np.average(putMotion)
    return([[call,put],[callMotion,putMotion]])

def AsianGeoFixSim(S,k,r,T):
    '''
    Use simulated underlying to determine the price of an Asian Geometric
    Average Call / Put option with a fixed strike price
    Payoffs are of the form :
    C = max(AVG_geo - K, 0)
    P = max(K - AVG_geo, 0)

    Parameters
    ----------
    S : numpy.array
        Simulated stock price, want to be of the form such that the first row
        is the initial stock price, with subsequent rows representing an
        additional time step increase, and each column is a simulated path of
        the asset
    k : number of any type (int, float8, float64 etc.)
        Strike value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.)
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.)
        Time till expiration for option

    Returns
    -------
    [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
        floats, second of one-dimensional numpy.array's
        First list is the call and put price, determined by the average
        of the simulated stock payoffs
        Second list is the call and put simulated paths payoffs at expiration,
        NOT discounted

    Notes
    -----
    The accuracy of pricing is dependent on the number of time steps and
    simulated paths chosen for the underlying stochastic motion

    '''
    avg = sctats.gmean(S,axis=0)
    callMotion = (avg - k).clip(0)
    putMotion = (k - avg).clip(0)

    call = np.exp(-r*T)*np.average(callMotion)
    put = np.exp(-r*T)*np.average(putMotion)
    return([[call,put],[callMotion,putMotion]])

def AsianGeoFloatSim(S,m,r,T):
    '''
    Use simulated underlying to determine the price of an Asian Geometric
    Average Call / Put option with a floating strike price
    Payoffs are of the form :
    C = max(S - m*AVG_geo, 0)
    P = max(m*AVG_geo - S, 0)

    Parameters
    ----------
    S : numpy.array
        Simulated stock price, want to be of the form such that the first row
        is the initial stock price, with subsequent rows representing an
        additional time step increase, and each column is a simulated path of
        the asset
    m : number of any type (int, float8, float64 etc.)
        Strike value scaler of option, determined at initiation
    r : number of any type (int, float8, float64 etc.)
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.)
        Time till expiration for option

    Returns
    -------
    [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
        floats, second of one-dimensional numpy.array's
        First list is the call and put price, determined by the average
        of the simulated stock payoffs
        Second list is the call and put simulated paths payoffs at expiration,
        NOT discounted

    Notes
    -----
    The accuracy of pricing is dependent on the number of time steps and
    simulated paths chosen for the underlying stochastic motion

    '''
    avg = sctats.gmean(S,axis=0)
    callMotion = (S[-1] - m*avg).clip(0)
    putMotion = (m*avg - S[-1]).clip(0)

    call = np.exp(-r*T)*np.average(callMotion)
    put = np.exp(-r*T)*np.average(putMotion)
    return([[call,put],[callMotion,putMotion]])

def AsianArithFixSim(S,k,r,T):
    '''
    Use simulated underlying to determine the price of an Asian Arithmetic
    Average Call / Put option with a fixed strike price
    Payoffs are of the form :
    C = max(AVG_arithmetic - K, 0)
    P = max(K - AVG_arithmetic, 0)

    Parameters
    ----------
    S : numpy.array
        Simulated stock price, want to be of the form such that the first row
        is the initial stock price, with subsequent rows representing an
        additional time step increase, and each column is a simulated path of
        the asset
    k : number of any type (int, float8, float64 etc.)
        Strike value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.)
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.)
        Time till expiration for option

    Returns
    -------
    [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
        floats, second of one-dimensional numpy.array's
        First list is the call and put price, determined by the average
        of the simulated stock payoffs
        Second list is the call and put simulated paths payoffs at expiration,
        NOT discounted

    Notes
    -----
    The accuracy of pricing is dependent on the number of time steps and
    simulated paths chosen for the underlying stochastic motion

    '''
    avg = np.average(S,axis=0)
    callMotion = (avg - k).clip(0)
    putMotion = (k - avg).clip(0)

    call = np.exp(-r*T)*np.average(callMotion)
    put = np.exp(-r*T)*np.average(putMotion)
    return([[call,put],[callMotion,putMotion]])

def AsianArithFloatSim(S,m,r,T):
    '''
    Use simulated underlying to determine the price of an Asian Arithmetic
    Average Call / Put option with a floating strike price
    Payoffs are of the form :
    C = max(S - m*AVG_arithmetic, 0)
    P = max(m*AVG_arithmetic - S, 0)

    Parameters
    ----------
    S : numpy.array
        Simulated stock price, want to be of the form such that the first row
        is the initial stock price, with subsequent rows representing an
        additional time step increase, and each column is a simulated path of
        the asset
    m : number of any type (int, float8, float64 etc.)
        Strike value scaler of option, determined at initiation
    r : number of any type (int, float8, float64 etc.)
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.)
        Time till expiration for option

    Returns
    -------
    [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
        floats, second of one-dimensional numpy.array's
        First list is the call and put price, determined by the average
        of the simulated stock payoffs
        Second list is the call and put simulated paths payoffs at expiration,
        NOT discounted

    Notes
    -----
    The accuracy of pricing is dependent on the number of time steps and
    simulated paths chosen for the underlying stochastic motion

    '''
    avg = np.average(S,axis=0)
    callMotion = (S[-1] - m*avg).clip(0)
    putMotion = (m*avg - S[-1]).clip(0)

    # class
    call = np.exp(-r*T)*np.average(callMotion)
    put = np.exp(-r*T)*np.average(putMotion)
    return([[call,put],[callMotion,putMotion]])

def PowerSim(S,k,r,T,n):
    '''
    Use simulated underlying to determine the price of a Power Call / Put
    option with a fixed strike price
    Payoffs are of the form :
    C = max(S**n - K, 0)
    P = max(K - S**n, 0)

    Parameters
    ----------
    S : numpy.array
        Simulated stock price, want to be of the form such that the first row
        is the initial stock price, with subsequent rows representing an
        additional time step increase, and each column is a simulated path of
        the asset
    k : number of any type (int, float8, float64 etc.)
        Strike value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.)
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.)
        Time till expiration for option
    n : number of any type (int, float8, float64 etc.)
        Power the underlying is raised to at expiration

    Returns
    -------
    [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
        floats, second of one-dimensional numpy.array's
        First list is the call and put price, determined by the average
        of the simulated stock payoffs
        Second list is the call and put simulated paths payoffs at expiration,
        NOT discounted

    Notes
    -----
    The accuracy of pricing is dependent on the number of time steps and
    simulated paths chosen for the underlying stochastic motion

    '''
    power = np.power(S[-1],n)
    callMotion = (power - k).clip(0)
    putMotion = (k - power).clip(0)

    call = np.exp(-r*T)*np.average(callMotion)
    put = np.exp(-r*T)*np.average(putMotion)
    return([[call,put],[callMotion,putMotion]])

def PowerStrikeSim(S,k,r,T,n):
    '''
    Use simulated underlying to determine the price of a Power Call / Put
    option with a fixed strike price
    Payoffs are of the form :
    C = max(S**n - K**n, 0)
    P = max(K**n - S**n, 0)

    Parameters
    ----------
    S : numpy.array
        Simulated stock price, want to be of the form such that the first row
        is the initial stock price, with subsequent rows representing an
        additional time step increase, and each column is a simulated path of
        the asset
    k : number of any type (int, float8, float64 etc.)
        Strike value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.)
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.)
        Time till expiration for option
    n : number of any type (int, float8, float64 etc.)
        Power the underlying and strike are raised to at expiration

    Returns
    -------
    [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
        floats, second of one-dimensional numpy.array's
        First list is the call and put price, determined by the average
        of the simulated stock payoffs
        Second list is the call and put simulated paths payoffs at expiration,
        NOT discounted

    Notes
    -----
    The accuracy of pricing is dependent on the number of time steps and
    simulated paths chosen for the underlying stochastic motion

    '''
    powerS = np.power(S[-1],n)
    callMotion = (powerS - k**n).clip(0)
    putMotion = (k**n - powerS).clip(0)

    call = np.exp(-r*T)*np.average(callMotion)
    put = np.exp(-r*T)*np.average(putMotion)
    return([[call,put],[callMotion,putMotion]])

def AvgBarrierSim(S,Z,r,timeMatrix):
    '''
    Use simulated underlying to determine the price of an Average Barrier
    option, where the payoff is the arithmetic average of the underlying over
    the time t (t is time when the option hits the barrier), or
    T (time to expiration) if the barrier is never hit
    Payoff is of the form :
    P = AVG_arithmetic_t

    Parameters
    ----------
    S : numpy.array
        Simulated stock price, want to be of the form such that the first row
        is the initial stock price, with subsequent rows representing an
        additional time step increase, and each column is a simulated path of
        the asset
    Z : number of any type (int, float8, float64 etc.)
        Barrier value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.)
        Risk free interest rate, implied constant till expiration
    timeMatrix : numpy.matrix
        Matrix of time intervals, of the same dimensions as the S matrix,
        should have 'paths' number of columns, and rows that iterate between 0
        and T by dt

    Returns
    -------
    [price,payoffMotion] : list, first is float, second is
        one-dimensional numpy.array
        price, is estimated price of the option, determined by the average
        of the simulated stock payoffs
        payoffMotion is the simulated paths payoffs at expiration,
        NOT discounted

    Notes
    -----
    The accuracy of pricing is dependent on the number of time steps and
    simulated paths chosen for the underlying stochastic motion
    * if the barrier is equal to the initial spot price, the price and
    payoffMotion will both be equal to the spot price since underlying hits the
    barrier at initiation

    '''
    s0 = S[0][0]
    if s0 < Z: # below
        hitBarrier = np.cumprod(S < Z,axis=0)
    elif s0 > Z: # above
        hitBarrier = np.cumprod(S > Z,axis=0)
    else: # on barrier
        price = s0
        payoffMotion = S[0]
        return([price,payoffMotion])

    paymentTime = np.array(np.max(np.multiply(timeMatrix,hitBarrier),axis=0))
    payoffMotion = np.sum(np.multiply(hitBarrier,S),axis=0) / np.sum(hitBarrier,axis=0)
    price = np.average(np.exp(-r*paymentTime)*payoffMotion)
    return([price,payoffMotion])

def NoTouchSingleSim(S,Z,r,T,payoutScale):
    '''
    Use simulated underlying to determine the price of a No Touch Binary
    option, with a single direction having a barrier. If the barrier is hit
    prior to expiration, their is no payoff. If it isn't, the payoff has a
    scale prior to initiation.
    Payoff is of the form :
    P = (money down * payoutScale, 0)_Z

    Parameters
    ----------
    S : numpy.array
        Simulated stock price, want to be of the form such that the first row
        is the initial stock price, with subsequent rows representing an
        additional time step increase, and each column is a simulated path of
        the asset
    Z : number of any type (int, float8, float64 etc.)
        Barrier value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.)
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.)
        Time till expiration for option
    payoutScale : number of any type (int, float8, float64 etc.)
        Scale value of payoff, should be a percentage (e.g. 20% payoff should
        be 0.2 when input)

    Returns
    -------
    [price,payoffMotion] : list, first is float, second is
        one-dimensional numpy.array
        price, is estimated return on option per dollar put down,
        determined by the average of the simulated stock payoffs
        payoffMotion is the simulated paths payoffs at expiration,
        NOT discounted

    Notes
    -----
    The accuracy of pricing is dependent on the number of time steps and
    simulated paths chosen for the underlying stochastic motion
    * if the barrier is equal to the initial spot price, the price and
    payoffMotion will both be 0 since underlying hits the barrier at initiation

    '''
    s0 = S[0][0]
    if s0 < Z: # below
        hitBarrier = np.cumprod(S < Z,axis=0)
    elif s0 > Z: # above
        hitBarrier = np.cumprod(S > Z,axis=0)
    else: # on barrier
        price = 0.0
        payoffMotion = S[0]*0.0
        return([price,payoffMotion])

    payoffMotion = (1+payoutScale)*hitBarrier[-1]
    price = np.average(np.exp(-r*T)*payoffMotion)
    return([price,payoffMotion])

def NoTouchDoubleSim(S,Z1,Z2,r,T,payoutScale):
    '''
    Use simulated underlying to determine the price of a No Touch Binary
    option, with both directions having a barrier. If either barrier is hit
    prior to expiration, their is no payoff. If they aren't hit, the payoff
    has a scale prior to initiation.
    Payoff is of the form :
    P = (money down * payoutScale, 0)_Z1,Z2

    Parameters
    ----------
    S : numpy.array
        Simulated stock price, want to be of the form such that the first row
        is the initial stock price, with subsequent rows representing an
        additional time step increase, and each column is a simulated path of
        the asset
    Z1 : number of any type (int, float8, float64 etc.)
        First barrier value of option, determined at initiation
    Z2 : number of any type (int, float8, float64 etc.)
        Second barrier value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.)
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.)
        Time till expiration for option
    payoutScale : number of any type (int, float8, float64 etc.)
        Scale value of payoff, should be a percentage (e.g. 20% payoff should
        be 0.2 when input)

    Returns
    -------
    [price,payoffMotion] : list, first is float, second is
        one-dimensional numpy.array
        price, is estimated return on option per dollar put down,
        determined by the average of the simulated stock payoffs
        payoffMotion is the simulated paths payoffs at expiration,
        NOT discounted

    Notes
    -----
    The accuracy of pricing is dependent on the number of time steps and
    simulated paths chosen for the underlying stochastic motion
    * if either barrier is equal to the initial spot price, the price and
    payoffMotion will both be 0 since underlying hits the barrier at initiation
    ** if spot is not between Z1 and Z2, then output error, since two
    barriers will be redundant

    '''
    s0 = S[0][0]
    if s0 < Z1 and s0 > Z2:
        hitBarrier1 = np.cumprod(S < Z1,axis=0)
        hitBarrier2 = np.cumprod(S > Z2,axis=0)
    elif s0 > Z1 and s0 < Z2:
        hitBarrier1 = np.cumprod(S > Z1,axis=0)
        hitBarrier2 = np.cumprod(S < Z2,axis=0)
    elif s0 == Z1 or s0 == Z2:
        price = 0.0
        payoffMotion = S[0]*0.0
        return([price,payoffMotion])
    else:
        print('Error : s0 outside barriers, use NoTouchSingle instead')
        return

    hitBarrier = np.multiply(hitBarrier1,hitBarrier2)
    payoffMotion = (1+payoutScale)*hitBarrier[-1]
    price = np.average(np.exp(-r*T)*payoffMotion)
    return([price,payoffMotion])

def CashOrNothingSim(S,Z,r,T,payout):
    '''
    Use simulated underlying to determine the price of a Cash-or-Nothing
    option, with a single direction having a barrier. If the barrier is hit
    prior to expiration, their is no payoff. If it isn't, the payoff is a
    value determined prior to initiation.
    Payoff is of the form :
    P = (payout, 0)_Z

    Parameters
    ----------
    S : numpy.array
        Simulated stock price, want to be of the form such that the first row
        is the initial stock price, with subsequent rows representing an
        additional time step increase, and each column is a simulated path of
        the asset
    Z : number of any type (int, float8, float64 etc.)
        Barrier value of option, determined at initiation
    r : number of any type (int, float8, float64 etc.)
        Risk free interest rate, implied constant till expiration
    T : number of any type (int, float8, float64 etc.)
        Time till expiration for option
    payout : number of any type (int, float8, float64 etc.)
        Payout of option, fixed value paid out if the barrier isn't hit by the
        underlying over the life of the option

    Returns
    -------
    [price,payoffMotion] : list, first is float, second is
        one-dimensional numpy.array
        price, is estimated price of the option, determined by the average
        of the simulated stock payoffs
        payoffMotion is the simulated paths payoffs at expiration,
        NOT discounted

    Notes
    -----
    The accuracy of pricing is dependent on the number of time steps and
    simulated paths chosen for the underlying stochastic motion
    * if the barrier is equal to the initial spot price, the price and
    payoffMotion will both be 0 since underlying hits the barrier at initiation

    '''
    s0 = S[0][0]
    if s0 < Z: # below
        hitBarrier = np.cumprod(S < Z,axis=0)
    elif s0 > Z: # above
        hitBarrier = np.cumprod(S > Z,axis=0)
    else: # on barrier
        price = 0.0
        payoffMotion = S[0]*0.0
        return([price,payoffMotion])

    payoffMotion = hitBarrier[-1]*payout
    price = np.average(np.exp(-r*T)*payoffMotion)
    return([price,payoffMotion])

# simulation functions
def SimpleSim(s0,r,T,vol,dt,paths):
    '''
    Simulate the motion of an underlying stock that follows a
    standard Weiner process for T/dt steps over a specified number of paths

    Parameters
    ----------
    s0 : number of any type (int, float8, float64 etc.)
        Spot value of underlying assets at current time, t
    r : number of any type (int, float8, float64 etc.)
        Risk free interest rate value
    T : number of any type (int, float8, float64 etc.)
        Time till expiration for option
    vol : number of any type (int, float8, float64 etc.)
        Volatility of underlying, implied constant till
        expiration in simple model
    dt : number of any type (int, float8, float64 etc.)
        Time interval used in simulation, used for number
        of stepsstock has, and motion of Weiner process
    paths : number of type 'int'
        Number of stocks simulated, higher number of paths
        leads to greater accuracy in calculated price.

    Notes
    -----
    * numpy matrices will begin to break once a large enough path is chosen
    ** similarly, if too small of a dt is chosen, the numpy matrices
    will begin to break

    Returns
    -------
    S : numpy.array
        A (T/dt) x paths array that holds the simulated
        stock price values

        of the form:
        [[s_0 s_0 ... s_0],
        [s_11 s_12 ... s_1paths]
        ...
        [s_(T/dt)1 s_(T/dt)2 ... s_(T/dt)paths]]

    '''
    intervals = int(T/dt)

    S = np.random.random([intervals+1,paths])
    S = -1 + 2*(S > 0.5)
    S = S*np.sqrt(dt)*vol + (r - 0.5*vol*vol)*dt
    S[0] = np.ones(paths)*np.log(s0)
    S = np.exp(np.matrix.cumsum(S,axis=0))
    return(S)

def HestonSim(s0,r,T,vol,phi,kappa,xi,dt,paths):
    '''
    Simulate the motion of an underlying stock that follows a
    standard Weiner process for T/dt steps over a specified number
    of paths with a stochastic volatility

    Parameters
    ----------
    s0 : number of any type (int, float8, float64 etc.)
        Spot value of underlying assets at current time, t
    r : number of any type (int, float8, float64 etc.)
        Risk free interest rate value
    T : number of any type (int, float8, float64 etc.)
        Time till expiration for option
    vol : number of any type (int, float8, float64 etc.)
        Volatility of underlying, implied constant till
        expiration in simple model
    phi : number of any type (int, float8, float64 etc.)
        Correlation of price to volatility
    kappa : number of any type (int, float8, float64 etc.)
        Speed of volatility adjustment
    xi : number of any type (int, float8, float64 etc.)
        Volatility of the volatility
    dt : number of any type (int, float8, float64 etc.)
        Time interval used in simulation, used for number
        of stepsstock has, and motion of Weiner process
    paths : number of type 'int'
        Number of stocks simulated, higher number of paths
        leads to greater accuracy in calculated price.

    Notes
    -----
    * numpy matrices will begin to break once a large enough path is chosen
    ** similarly, if too small of a dt is chosen, the numpy matrices
    will begin to break

    Returns
    -------
    [S, volMotion] : list of numpy.array's
        A (T/dt) x paths array that holds the simulated
        stock price values
        A (T/dt) x paths array that holds the simulated
        volatility motion

        both of the form:
        [[s_0 s_0 ... s_0],
        [s_11 s_12 ... s_1paths]
        ...
        [s_(T/dt)1 s_(T/dt)2 ... s_(T/dt)paths]]

    '''
    intervals = int(T/dt)

    S = np.sqrt(dt)*(-1 + 2*(np.random.random([intervals+1,paths]) > 0.5))
    V = np.sqrt(dt)*(-1 + 2*(np.random.random([intervals+1,paths]) > 0.5))
    V = phi*S + np.sqrt(1 - phi*phi)*V

    volMotion = np.zeros([intervals+1,paths])
    volMotion[0] = vol*vol*np.ones(paths)

    for t in range(intervals):
        vt = volMotion[t]
        dvt = kappa*(vol*vol - vt)*dt + xi*np.sqrt(vt)*V[t]
        volMotion[t+1] = vt + dvt

    S = (r - 0.5*volMotion)*dt + np.sqrt(volMotion)*S
    S[0] = np.ones(paths)*np.log(s0)
    S = np.exp(np.matrix.cumsum(S,axis=0))
    return([S, volMotion])

# classes utilizing the different simulation and payoff functions
class Simple:
    '''
    This is a simple risk neutral simulation class.

    Simulate the motion of an underlying stock that follows a standard
    Weiner process for T/dt steps over a specified number of paths.

    Determine the pricing of various options using the simulated stochastic
    motion of the underlying asset.

    '''
    def __init__(self,s0,r,T,vol,dt=0.001,paths=1000):
        '''
        Parameters
        ----------
        s0 : number of any type (int, float8, float64 etc.)
            Spot value of underlying assets at current time, t
        r : number of any type (int, float8, float64 etc.)
            Risk free interest rate value
        T : number of any type (int, float8, float64 etc.)
            Time till expiration for option
        vol : number of any type (int, float8, float64 etc.)
            Volatility of underlying, implied constant till
            expiration in simple model
        dt : number of any type (int, float8, float64 etc.)
            Time interval used in simulation, used for number
            of stepsstock has, and motion of Weiner process
            Standard is 0.001, unless changed by user
        paths : number of type 'int'
            Number of stocks simulated, higher number of paths
            leads to greater accuracy in calculated price.
            Standard is 10000, unless changed by user

        Notes
        -----
        * numpy matrices will begin to break once a large enough path is chosen
        ** similarly, if too small of a dt is chosen, the numpy matrices
        will begin to break

        Self
        ----
        timeMatrix : numpy.array
            A (T/dt) x paths array that holds all the time values, increasing
            from zero to time T
        S : numpy.array
            A (T/dt) x paths array that holds the simulated
            stock price values
        simtime : the time to complete the simulation, useful
            in testing efficiency for variable paths and dt

        '''
        self.s0 = s0
        self.r = r
        self.T = T
        self.vol = vol
        self.dt = dt
        self.paths = paths

        timeInt = np.matrix(np.arange(0,T+dt,dt)).transpose()
        self.timeMatrix = np.matmul(timeInt,np.matrix(np.ones(paths)))

        start = time.time()
        self.S = SimpleSim(s0,r,T,vol,dt,paths)
        self.simtime = time.time() - start

    def Euro(self,k):
        '''
        Use simulated underlying to determine the price of a European
        Call / Put option
        Payoffs are of the form :
        C = max(S - K, 0)
        P = max(K - S, 0)

        Parameters
        ----------
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation

        Returns
        -------
        [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
            floats, second of one-dimensional numpy.array's
            First list is the call and put price, determined by the average
            of the simulated stock payoffs
            Second list is the call and put simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion

        '''
        out = EuroSim(self.S,k,self.r,self.T)
        return(out)

    def AsianGeoFix(self,k):
        '''
        Use simulated underlying to determine the price of an Asian Geometric
        Average Call / Put option with a fixed strike price
        Payoffs are of the form :
        C = max(AVG_geo - K, 0)
        P = max(K - AVG_geo, 0)

        Parameters
        ----------
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation

        Returns
        -------
        [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
            floats, second of one-dimensional numpy.array's
            First list is the call and put price, determined by the average
            of the simulated stock payoffs
            Second list is the call and put simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion

        '''
        out = AsianGeoFixSim(self.S,k,self.r,self.T)
        return(out)

    def AsianArithFix(self,k):
        '''
        Use simulated underlying to determine the price of an Asian Arithmetic
        Average Call / Put option with a fixed strike price
        Payoffs are of the form :
        C = max(AVG_arithmetic - K, 0)
        P = max(K - AVG_arithmetic, 0)

        Parameters
        ----------
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation

        Returns
        -------
        [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
            floats, second of one-dimensional numpy.array's
            First list is the call and put price, determined by the average
            of the simulated stock payoffs
            Second list is the call and put simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion

        '''
        out = AsianArithFixSim(self.S,k,self.r,self.T)
        return(out)

    def AsianGeoFloat(self,m):
        '''
        Use simulated underlying to determine the price of an Asian Geometric
        Average Call / Put option with a floating strike price
        Payoffs are of the form :
        C = max(S - m*AVG_geo, 0)
        P = max(m*AVG_geo - S, 0)

        Parameters
        ----------
        m : number of any type (int, float8, float64 etc.)
            Strike value scaler of option, determined at initiation

        Returns
        -------
        [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
            floats, second of one-dimensional numpy.array's
            First list is the call and put price, determined by the average
            of the simulated stock payoffs
            Second list is the call and put simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion

        '''
        out = AsianGeoFloatSim(self.S,m,self.r,self.T)
        return(out)

    def AsianArithFloat(self,m):
        '''
        Use simulated underlying to determine the price of an Asian Arithmetic
        Average Call / Put option with a floating strike price
        Payoffs are of the form :
        C = max(S - m*AVG_arithmetic, 0)
        P = max(m*AVG_arithmetic - S, 0)

        Parameters
        ----------
        m : number of any type (int, float8, float64 etc.)
            Strike value scaler of option, determined at initiation

        Returns
        -------
        [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
            floats, second of one-dimensional numpy.array's
            First list is the call and put price, determined by the average
            of the simulated stock payoffs
            Second list is the call and put simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion

        '''
        out = AsianArithFloatSim(self.S,m,self.r,self.T)
        return(out)

    def Power(self,k,n):
        '''
        Use simulated underlying to determine the price of a Power Call / Put
        option with a fixed strike price
        Payoffs are of the form :
        C = max(S**n - K, 0)
        P = max(K - S**n, 0)

        Parameters
        ----------
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation
        n : number of any type (int, float8, float64 etc.)
            Power the underlying is raised to at expiration


        Returns
        -------
        [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
            floats, second of one-dimensional numpy.array's
            First list is the call and put price, determined by the average
            of the simulated stock payoffs
            Second list is the call and put simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion

        '''
        out = PowerSim(self.S,k,self.r,self.T,n)
        return(out)

    def PowerStrike(self,k,n):
        '''
        Use simulated underlying to determine the price of a Power Call / Put
        option with a fixed strike price
        Payoffs are of the form :
        C = max(S**n - K**n, 0)
        P = max(K**n - S**n, 0)

        Parameters
        ----------
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation
        n : number of any type (int, float8, float64 etc.)
            Power the underlying and strike are raised to at expiration

        Returns
        -------
        [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
            floats, second of one-dimensional numpy.array's
            First list is the call and put price, determined by the average
            of the simulated stock payoffs
            Second list is the call and put simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion

        '''
        out = PowerStrikeSim(self.S,k,self.r,self.T,n)
        return(out)

    def AvgBarrier(self,Z):
        '''
        Use simulated underlying to determine the price of an Average Barrier
        option, where the payoff is the arithmetic average of the underlying over
        the time t (t is time when the option hits the barrier), or
        T (time to expiration) if the barrier is never hit
        Payoff is of the form :
        P = AVG_arithmetic_t

        Parameters
        ----------
        Z : number of any type (int, float8, float64 etc.)
            Barrier value of option, determined at initiation

        Returns
        -------
        [price,payoffMotion] : list, first is float, second is
            one-dimensional numpy.array
            price, is estimated price of the option, determined by the average
            of the simulated stock payoffs
            payoffMotion is the simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion
        * if the barrier is equal to the initial spot price, the price and
        payoffMotion will both be equal to the spot price since underlying hits the
        barrier at initiation

        '''
        out = AvgBarrierSim(self.S,Z,self.r,self.timeMatrix)
        return(out)

    def NoTouchSingle(self,Z,payoutScale):
        '''
        Use simulated underlying to determine the price of a No Touch Binary
        option, with a single direction having a barrier. If the barrier is hit
        prior to expiration, their is no payoff. If it isn't, the payoff has a
        scale prior to initiation.
        Payoff is of the form :
        P = (money down * payoutScale, 0)_Z

        Parameters
        ----------
        Z : number of any type (int, float8, float64 etc.)
            Barrier value of option, determined at initiation
        payoutScale : number of any type (int, float8, float64 etc.)
            Scale value of payoff, should be a percentage (e.g. 20% payoff should
            be 0.2 when input)

        Returns
        -------
        [price,payoffMotion] : list, first is float, second is
            one-dimensional numpy.array
            price, is estimated return on option per dollar put down,
            determined by the average of the simulated stock payoffs
            payoffMotion is the simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion
        * if the barrier is equal to the initial spot price, the price and
        payoffMotion will both be 0 since underlying hits the barrier at initiation

        '''
        out = NoTouchSingleSim(self.S,Z,self.r,self.T,payoutScale)
        return(out)

    def NoTouchDouble(self,Z1,Z2,payoutScale):
        '''
        Use simulated underlying to determine the price of a No Touch Binary
        option, with both directions having a barrier. If either barrier is hit
        prior to expiration, their is no payoff. If they aren't hit, the payoff
        has a scale prior to initiation.
        Payoff is of the form :
        P = (money down * payoutScale, 0)_Z1,Z2

        Parameters
        ----------
        Z1 : number of any type (int, float8, float64 etc.)
            First barrier value of option, determined at initiation
        Z2 : number of any type (int, float8, float64 etc.)
            Second barrier value of option, determined at initiation
        payoutScale : number of any type (int, float8, float64 etc.)
            Scale value of payoff, should be a percentage (e.g. 20% payoff should
            be 0.2 when input)

        Returns
        -------
        [price,payoffMotion] : list, first is float, second is
            one-dimensional numpy.array
            price, is estimated return on option per dollar put down,
            determined by the average of the simulated stock payoffs
            payoffMotion is the simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion
        * if either barrier is equal to the initial spot price, the price and
        payoffMotion will both be 0 since underlying hits the barrier at initiation
        ** if spot is not between Z1 and Z2, then output error, since two
        barriers will be redundant

        '''
        out = NoTouchDoubleSim(self.S,Z1,Z2,self.r,self.T,payoutScale)
        return(out)

    def CashOrNothing(self,Z,payout):
        '''
        Use simulated underlying to determine the price of a Cash-or-Nothing
        option, with a single direction having a barrier. If the barrier is hit
        prior to expiration, their is no payoff. If it isn't, the payoff is a
        value determined prior to initiation.
        Payoff is of the form :
        P = (payout, 0)_Z

        Parameters
        ----------
        Z : number of any type (int, float8, float64 etc.)
            Barrier value of option, determined at initiation
        payout : number of any type (int, float8, float64 etc.)
            Payout of option, fixed value paid out if the barrier isn't hit by the
            underlying over the life of the option

        Returns
        -------
        [price,payoffMotion] : list, first is float, second is
            one-dimensional numpy.array
            price, is estimated price of the option, determined by the average
            of the simulated stock payoffs
            payoffMotion is the simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion
        * if the barrier is equal to the initial spot price, the price and
        payoffMotion will both be 0 since underlying hits the barrier at initiation

        '''
        out = CashOrNothingSim(self.S,Z,self.r,self.T,payout)
        return(out)

class Heston:
    '''
    This is a Heston model simulation class.

    Simulate the motion of an underlying stock that follows a standard
    Weiner process for T/dt steps over a specified number of paths,
    with stochastic volatility.

    Determine the pricing of various options using the simulated stochastic
    motion of the underlying asset.

    '''
    def __init__(self,s0,r,T,vol,phi,kappa,xi,dt=0.001,paths=1000):
        '''
        Parameters
        ----------
        s0 : number of any type (int, float8, float64 etc.)
            Spot value of underlying assets at current time, t
        r : number of any type (int, float8, float64 etc.)
            Risk free interest rate value
        T : number of any type (int, float8, float64 etc.)
            Time till expiration for option
        vol : number of any type (int, float8, float64 etc.)
            Volatility of underlying, implied constant till
            expiration in simple model
        phi : number of any type (int, float8, float64 etc.)
            Correlation of price to volatility
        kappa : number of any type (int, float8, float64 etc.)
            Speed of volatility adjustment
        xi : number of any type (int, float8, float64 etc.)
            Volatility of the volatility
        dt : number of any type (int, float8, float64 etc.)
            Time interval used in simulation, used for number
            of stepsstock has, and motion of Weiner process
            Standard is 0.001, unless changed by user
        paths : number of type 'int'
            Number of stocks simulated, higher number of paths
            leads to greater accuracy in calculated price.
            Standard is 10000, unless changed by user

        Notes
        -----
        * numpy matrices will begin to break once a large enough path is chosen
        ** similarly, if too small of a dt is chosen, the numpy matrices
        will begin to break

        Self
        ----
        timeMatrix : numpy.array
            A (T/dt) x paths array that holds all the time values, increasing
            from zero to time T
        S : numpy.array
            A (T/dt) x paths array that holds the simulated
            stock price values
        volMotion : numpy.array
            A (T/dt) x paths array that holds the simulated
            volatility motion
        simtime : the time to complete the simulation, useful
            in testing efficiency for variable paths and dt

        '''
        self.s0 = s0
        self.r = r
        self.T = T
        self.vol = vol
        self.dt = dt
        self.paths = paths
        self.phi = phi # correlation of price to volatility
        self.kappa = kappa # speed of adjustment
        self.xi = xi # volatility of volatility

        timeInt = np.matrix(np.arange(0,T+dt,dt)).transpose()
        self.timeMatrix = np.matmul(timeInt,np.matrix(np.ones(paths)))

        start = time.time()
        [self.S, self.volMotion] = HestonSim(s0,r,vol,phi,kappa,xi,dt,paths)
        self.simtime = time.time() - start

    def Euro(self,k):
        '''
        Use simulated underlying to determine the price of a European
        Call / Put option
        Payoffs are of the form :
        C = max(S - K, 0)
        P = max(K - S, 0)

        Parameters
        ----------
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation

        Returns
        -------
        [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
            floats, second of one-dimensional numpy.array's
            First list is the call and put price, determined by the average
            of the simulated stock payoffs
            Second list is the call and put simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion

        '''
        out = EuroSim(self.S,k,self.r,self.T)
        return(out)

    def AsianGeoFix(self,k):
        '''
        Use simulated underlying to determine the price of an Asian Geometric
        Average Call / Put option with a fixed strike price
        Payoffs are of the form :
        C = max(AVG_geo - K, 0)
        P = max(K - AVG_geo, 0)

        Parameters
        ----------
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation

        Returns
        -------
        [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
            floats, second of one-dimensional numpy.array's
            First list is the call and put price, determined by the average
            of the simulated stock payoffs
            Second list is the call and put simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion

        '''
        out = AsianGeoFixSim(self.S,k,self.r,self.T)
        return(out)

    def AsianArithFix(self,k):
        '''
        Use simulated underlying to determine the price of an Asian Arithmetic
        Average Call / Put option with a fixed strike price
        Payoffs are of the form :
        C = max(AVG_arithmetic - K, 0)
        P = max(K - AVG_arithmetic, 0)

        Parameters
        ----------
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation

        Returns
        -------
        [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
            floats, second of one-dimensional numpy.array's
            First list is the call and put price, determined by the average
            of the simulated stock payoffs
            Second list is the call and put simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion

        '''
        out = AsianArithFixSim(self.S,k,self.r,self.T)
        return(out)

    def AsianGeoFloat(self,m):
        '''
        Use simulated underlying to determine the price of an Asian Geometric
        Average Call / Put option with a floating strike price
        Payoffs are of the form :
        C = max(S - m*AVG_geo, 0)
        P = max(m*AVG_geo - S, 0)

        Parameters
        ----------
        m : number of any type (int, float8, float64 etc.)
            Strike value scaler of option, determined at initiation

        Returns
        -------
        [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
            floats, second of one-dimensional numpy.array's
            First list is the call and put price, determined by the average
            of the simulated stock payoffs
            Second list is the call and put simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion

        '''
        out = AsianGeoFloatSim(self.S,m,self.r,self.T)
        return(out)

    def AsianArithFloat(self,m):
        '''
        Use simulated underlying to determine the price of an Asian Arithmetic
        Average Call / Put option with a floating strike price
        Payoffs are of the form :
        C = max(S - m*AVG_arithmetic, 0)
        P = max(m*AVG_arithmetic - S, 0)

        Parameters
        ----------
        m : number of any type (int, float8, float64 etc.)
            Strike value scaler of option, determined at initiation

        Returns
        -------
        [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
            floats, second of one-dimensional numpy.array's
            First list is the call and put price, determined by the average
            of the simulated stock payoffs
            Second list is the call and put simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion

        '''
        out = AsianArithFloatSim(self.S,m,self.r,self.T)
        return(out)

    def Power(self,k,n):
        '''
        Use simulated underlying to determine the price of a Power Call / Put
        option with a fixed strike price
        Payoffs are of the form :
        C = max(S**n - K, 0)
        P = max(K - S**n, 0)

        Parameters
        ----------
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation
        n : number of any type (int, float8, float64 etc.)
            Power the underlying is raised to at expiration


        Returns
        -------
        [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
            floats, second of one-dimensional numpy.array's
            First list is the call and put price, determined by the average
            of the simulated stock payoffs
            Second list is the call and put simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion

        '''
        out = PowerSim(self.S,k,self.r,self.T,n)
        return(out)

    def PowerStrike(self,k,n):
        '''
        Use simulated underlying to determine the price of a Power Call / Put
        option with a fixed strike price
        Payoffs are of the form :
        C = max(S**n - K**n, 0)
        P = max(K**n - S**n, 0)

        Parameters
        ----------
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation
        n : number of any type (int, float8, float64 etc.)
            Power the underlying and strike are raised to at expiration

        Returns
        -------
        [[call,put],[callMotion,putMotion]] : list of pair of lists, first of
            floats, second of one-dimensional numpy.array's
            First list is the call and put price, determined by the average
            of the simulated stock payoffs
            Second list is the call and put simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion

        '''
        out = PowerStrikeSim(self.S,k,self.r,self.T,n)
        return(out)

    def AvgBarrier(self,Z):
        '''
        Use simulated underlying to determine the price of an Average Barrier
        option, where the payoff is the arithmetic average of the underlying over
        the time t (t is time when the option hits the barrier), or
        T (time to expiration) if the barrier is never hit
        Payoff is of the form :
        P = AVG_arithmetic_t

        Parameters
        ----------
        Z : number of any type (int, float8, float64 etc.)
            Barrier value of option, determined at initiation

        Returns
        -------
        [price,payoffMotion] : list, first is float, second is
            one-dimensional numpy.array
            price, is estimated price of the option, determined by the average
            of the simulated stock payoffs
            payoffMotion is the simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion
        * if the barrier is equal to the initial spot price, the price and
        payoffMotion will both be equal to the spot price since underlying hits the
        barrier at initiation

        '''
        out = AvgBarrierSim(self.S,Z,self.r,self.timeMatrix)
        return(out)

    def NoTouchSingle(self,Z,payoutScale):
        '''
        Use simulated underlying to determine the price of a No Touch Binary
        option, with a single direction having a barrier. If the barrier is hit
        prior to expiration, their is no payoff. If it isn't, the payoff has a
        scale prior to initiation.
        Payoff is of the form :
        P = (money down * payoutScale, 0)_Z

        Parameters
        ----------
        Z : number of any type (int, float8, float64 etc.)
            Barrier value of option, determined at initiation
        payoutScale : number of any type (int, float8, float64 etc.)
            Scale value of payoff, should be a percentage (e.g. 20% payoff should
            be 0.2 when input)

        Returns
        -------
        [price,payoffMotion] : list, first is float, second is
            one-dimensional numpy.array
            price, is estimated return on option per dollar put down,
            determined by the average of the simulated stock payoffs
            payoffMotion is the simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion
        * if the barrier is equal to the initial spot price, the price and
        payoffMotion will both be 0 since underlying hits the barrier at initiation

        '''
        out = NoTouchSingleSim(self.S,Z,self.r,self.T,payoutScale)
        return(out)

    def NoTouchDouble(self,Z1,Z2,payoutScale):
        '''
        Use simulated underlying to determine the price of a No Touch Binary
        option, with both directions having a barrier. If either barrier is hit
        prior to expiration, their is no payoff. If they aren't hit, the payoff
        has a scale prior to initiation.
        Payoff is of the form :
        P = (money down * payoutScale, 0)_Z1,Z2

        Parameters
        ----------
        Z1 : number of any type (int, float8, float64 etc.)
            First barrier value of option, determined at initiation
        Z2 : number of any type (int, float8, float64 etc.)
            Second barrier value of option, determined at initiation
        payoutScale : number of any type (int, float8, float64 etc.)
            Scale value of payoff, should be a percentage (e.g. 20% payoff should
            be 0.2 when input)

        Returns
        -------
        [price,payoffMotion] : list, first is float, second is
            one-dimensional numpy.array
            price, is estimated return on option per dollar put down,
            determined by the average of the simulated stock payoffs
            payoffMotion is the simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion
        * if either barrier is equal to the initial spot price, the price and
        payoffMotion will both be 0 since underlying hits the barrier at initiation
        ** if spot is not between Z1 and Z2, then output error, since two
        barriers will be redundant

        '''
        out = NoTouchDoubleSim(self.S,Z1,Z2,self.r,self.T,payoutScale)
        return(out)

    def CashOrNothing(self,Z,payout):
        '''
        Use simulated underlying to determine the price of a Cash-or-Nothing
        option, with a single direction having a barrier. If the barrier is hit
        prior to expiration, their is no payoff. If it isn't, the payoff is a
        value determined prior to initiation.
        Payoff is of the form :
        P = (payout, 0)_Z

        Parameters
        ----------
        Z : number of any type (int, float8, float64 etc.)
            Barrier value of option, determined at initiation
        payout : number of any type (int, float8, float64 etc.)
            Payout of option, fixed value paid out if the barrier isn't hit by the
            underlying over the life of the option

        Returns
        -------
        [price,payoffMotion] : list, first is float, second is
            one-dimensional numpy.array
            price, is estimated price of the option, determined by the average
            of the simulated stock payoffs
            payoffMotion is the simulated paths payoffs at expiration,
            NOT discounted

        Notes
        -----
        The accuracy of pricing is dependent on the number of time steps and
        simulated paths chosen for the underlying stochastic motion
        * if the barrier is equal to the initial spot price, the price and
        payoffMotion will both be 0 since underlying hits the barrier at initiation

        '''
        out = CashOrNothingSim(self.S,Z,self.r,self.T,payout)
        return(out)

