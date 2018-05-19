# Austin Griffith
# Simulation

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

    * the accuracy of pricing is dependent on the number of time steps and
    simulated paths chosen for the underlying stochastic motion

    Examples
    --------
    >>> from simulation import EuroSim
    >>> import numpy as np
    >>> s0 = 1
        r = 0.015
        T = 0.5
        vol = 0.25
        dt = 0.1
        paths = 5
        k = 0.8

    >>> S = np.array([[ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
               [ 0.92248705,  1.08050869,  0.92248705,  1.08050869,  0.92248705],
               [ 0.85098236,  0.99675528,  0.85098236,  0.99675528,  0.99675528],
               [ 0.91949383,  0.91949383,  0.91949383,  1.07700274,  0.91949383],
               [ 0.99352108,  0.99352108,  0.84822115,  1.16371082,  0.99352108],
               [ 0.91651033,  1.07350816,  0.78247303,  1.07350816,  1.07350816]])
    >>> a = EuroSim(S,k,r,T)
    >>> print(a[0])
        print(a[1][0])
        print(a[1][1])
        [0.1860066674534242, 0.0034792018881946852]
        [ 0.11651033  0.27350816  0.          0.27350816  0.27350816]
        [ 0.          0.          0.01752697  0.          0.        ]

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

    * the accuracy of pricing is dependent on the number of time steps and
    simulated paths chosen for the underlying stochastic motion

    Examples
    --------
    >>> from simulation import AsianGeoFixSim
    >>> import numpy as np
    >>> s0 = 1
        r = 0.015
        T = 0.5
        vol = 0.25
        dt = 0.1
        paths = 5
        k = 0.8

    >>> S = np.array([[ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
               [ 0.92248705,  1.08050869,  0.92248705,  1.08050869,  0.92248705],
               [ 0.85098236,  0.99675528,  0.85098236,  0.99675528,  0.99675528],
               [ 0.91949383,  0.91949383,  0.91949383,  1.07700274,  0.91949383],
               [ 0.99352108,  0.99352108,  0.84822115,  1.16371082,  0.99352108],
               [ 0.91651033,  1.07350816,  0.78247303,  1.07350816,  1.07350816]])
    >>> a = AsianGeoFixSim(S,k,r,T)
    >>> print(a[0])
        print(a[1][0])
        print(a[1][1])
        [0.17326664943627884, 0.0]
        [ 0.1324467   0.20915531  0.08457506  0.26376902  0.18290908]
        [ 0.  0.  0.  0.  0.]

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

    * the accuracy of pricing is dependent on the number of time steps and
    simulated paths chosen for the underlying stochastic motion

    Examples
    --------
    >>> from simulation import AsianGeoFloatSim
    >>> import numpy as np
    >>> s0 = 1
        r = 0.015
        T = 0.5
        vol = 0.25
        dt = 0.1
        paths = 5
        m = 0.8

    >>> S = np.array([[ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
               [ 0.92248705,  1.08050869,  0.92248705,  1.08050869,  0.92248705],
               [ 0.85098236,  0.99675528,  0.85098236,  0.99675528,  0.99675528],
               [ 0.91949383,  0.91949383,  0.91949383,  1.07700274,  0.91949383],
               [ 0.99352108,  0.99352108,  0.84822115,  1.16371082,  0.99352108],
               [ 0.91651033,  1.07350816,  0.78247303,  1.07350816,  1.07350816]])
    >>> a = AsianGeoFloatSim(S,k,r,T)
    >>> print(a[0])
        print(a[1][0])
        print(a[1][1])
        [0.20271863478726854, 0.0]
        [ 0.17055297  0.26618391  0.07481299  0.22249294  0.2871809 ]
        [ 0.  0.  0.  0.  0.]

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

    * the accuracy of pricing is dependent on the number of time steps and
    simulated paths chosen for the underlying stochastic motion

    Examples
    --------
    >>> from simulation import AsianArithFixSim
    >>> import numpy as np
    >>> s0 = 1
        r = 0.015
        T = 0.5
        vol = 0.25
        dt = 0.1
        paths = 5
        k = 0.8

    >>> S = np.array([[ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
               [ 0.92248705,  1.08050869,  0.92248705,  1.08050869,  0.92248705],
               [ 0.85098236,  0.99675528,  0.85098236,  0.99675528,  0.99675528],
               [ 0.91949383,  0.91949383,  0.91949383,  1.07700274,  0.91949383],
               [ 0.99352108,  0.99352108,  0.84822115,  1.16371082,  0.99352108],
               [ 0.91651033,  1.07350816,  0.78247303,  1.07350816,  1.07350816]])
    >>> a = AsianArithFixSim(S,k,r,T)
    >>> print(a[0])
        print(a[1][0])
        print(a[1][1])
        [0.17493936228974066, 0.0]
        [ 0.13383244  0.21063117  0.08727624  0.26524762  0.18429423]
        [ 0.  0.  0.  0.  0.]

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

    * the accuracy of pricing is dependent on the number of time steps and
    simulated paths chosen for the underlying stochastic motion

    Examples
    --------
    >>> from simulation import AsianArithFloatSim
    >>> import numpy as np
    >>> s0 = 1
        r = 0.015
        T = 0.5
        vol = 0.25
        dt = 0.1
        paths = 5
        m = 0.8

    >>> S = np.array([[ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
               [ 0.92248705,  1.08050869,  0.92248705,  1.08050869,  0.92248705],
               [ 0.85098236,  0.99675528,  0.85098236,  0.99675528,  0.99675528],
               [ 0.91949383,  0.91949383,  0.91949383,  1.07700274,  0.91949383],
               [ 0.99352108,  0.99352108,  0.84822115,  1.16371082,  0.99352108],
               [ 0.91651033,  1.07350816,  0.78247303,  1.07350816,  1.07350816]])
    >>> a = AsianArithFloatSim(S,k,r,T)
    >>> print(a[0])
        print(a[1][0])
        print(a[1][1])
        [0.20138046450449909, 0.0]
        [ 0.16944438  0.26500322  0.07265204  0.22131007  0.28607277]
        [ 0.  0.  0.  0.  0.]

    '''
    avg = np.average(S,axis=0)
    callMotion = (S[-1] - m*avg).clip(0)
    putMotion = (m*avg - S[-1]).clip(0)

    # class
    call = np.exp(-r*T)*np.average(callMotion)
    put = np.exp(-r*T)*np.average(putMotion)
    return([[call,put],[callMotion,putMotion]])

def PowerSim(S,k,r,T,n):
    power = np.power(S[-1],n)
    callMotion = (power - k).clip(0)
    putMotion = (k - power).clip(0)

    call = np.exp(-r*T)*np.average(callMotion)
    put = np.exp(-r*T)*np.average(putMotion)
    return([[call,put],[callMotion,putMotion]])

def PowerStrikeSim(S,k,r,T,n):
    powerS = np.power(S[-1],n)
    callMotion = (powerS - k**n).clip(0)
    putMotion = (k**n - powerS).clip(0)

    call = np.exp(-r*T)*np.average(callMotion)
    put = np.exp(-r*T)*np.average(putMotion)
    return([[call,put],[callMotion,putMotion]])

def AvgBarrierSim(S,Z,r,timeMatrix):
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

def NoTouchSingleSim(S,Z,r,T,payoutScale):
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

    * numpy matrices will begin to break once a large enough path is chosen
    ** similarly, if too small of a dt is chosen, the numpy matrices
    will begin to break

    Returns
    -------
    S : numpy.array
        A (T/dt)**paths array that holds the simulated
        stock price values

        of the form:
        [[s_0 s_0 ... s_0],
        [s_11 s_12 ... s_1paths]
        ...
        [s_(T/dt)1 s_(T/dt)2 ... s_(T/dt)paths]]

    Examples
    --------
    >>> from qcfoptions.simulation import SimpleSim
    >>> s0 = 1
        r = 0.015
        T = 2
        vol = 0.25
        dt = 0.001
        paths = 1000
    >>> SimpleSim(s0,r,T,vol,dt,paths)
    array([[ 1.        ,  1.        ,  1.        , ...,  1.        ,
         1.        ,  1.        ],
       [ 1.00792065,  1.00792065,  1.00792065, ...,  1.00792065,
         0.99210935,  1.00792065],
       [ 1.01590403,  0.9999675 ,  1.01590403, ...,  1.01590403,
         0.9999675 ,  1.01590403],
       ...,
       [ 0.83965012,  0.96805391,  0.86662648, ...,  1.34928073,
         0.67291959,  0.36321019],
       [ 0.83302474,  0.97572153,  0.85978823, ...,  1.3599679 ,
         0.66760982,  0.36034423],
       [ 0.83962283,  0.98344987,  0.86659831, ...,  1.34923687,
         0.66234195,  0.36319839]])

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

    * numpy matrices will begin to break once a large enough path is chosen
    ** similarly, if too small of a dt is chosen, the numpy matrices
    will begin to break

    Returns
    -------
    [S, volMotion] : list of numpy.array's
        A (T/dt)**paths array that holds the simulated
        stock price values
        A (T/dt)**paths array that holds the simulated
        volatility motion

        both of the form:
        [[s_0 s_0 ... s_0],
        [s_11 s_12 ... s_1paths]
        ...
        [s_(T/dt)1 s_(T/dt)2 ... s_(T/dt)paths]]

    Examples
    --------
    >>> from qcfoptions.simulation import HestonSim
    >>> s0 = 1
        r = 0.015
        T = 2
        vol = 0.25
        vol = 0.25
        phi = -0.4
        kappa = 8
        dt = 0.001
        paths = 1000
    >>> HestonSim(s0,r,T,vol,dt,paths)
    [array([[ 1.        ,  1.        ,  1.        , ...,  1.        ,
          1.        ,  1.        ],
        [ 0.99220026,  0.99220026,  0.99220026, ...,  1.00768681,
          0.99188224,  0.99201948],
        [ 0.98455083,  0.98455083,  1.00019494, ...,  0.9998787 ,
          1.00018749,  0.98419354],
        ...,
        [ 0.57934074,  0.78773209,  0.77628209, ...,  0.67412693,
          0.94029086,  0.85260689],
        [ 0.58289171,  0.79231336,  0.78272252, ...,  0.67913971,
          0.93453208,  0.85863832],
        [ 0.57946344,  0.79675135,  0.77618405, ...,  0.68403361,
          0.92858262,  0.85250678]]),
 array([[ 0.0625    ,  0.0625    ,  0.0625    , ...,  0.0625    ,
          0.0625    ,  0.0625    ],
        [ 0.06107081,  0.06107081,  0.06107081, ...,  0.05885721,
          0.06614279,  0.06392919],
        [ 0.05966948,  0.05966948,  0.06468314, ...,  0.06027327,
          0.06986109,  0.06247232],
        ...,
        [ 0.03445569,  0.03621109,  0.07259113, ...,  0.05637955,
          0.03862534,  0.04846038],
        [ 0.03738478,  0.03364863,  0.06858453, ...,  0.0550711 ,
          0.0376928 ,  0.04983118],
        [ 0.03476835,  0.03120657,  0.070033  , ...,  0.05171108,
          0.0407202 ,  0.05120868]])]

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


class Simple:
    '''
    Simulate the motion of an underlying stock that follows a standard
    Weiner process for T/dt steps over a specified number of paths.

    Plot the underlying motion.

    Determine the pricing of various options using the simulated stochastic
    motion of the underlying asset.

    '''
    def __init__(self,s0,r,T,vol,dt=0.001,paths=10000):
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

        * numpy matrices will begin to break once a large enough path is chosen
        ** similarly, if too small of a dt is chosen, the numpy matrices
        will begin to break

        Self
        ----
        timeMatrix : numpy.array
            A (T/dt)**paths array that holds all the time values, increasing
            from zero to time T
        S : numpy.array
            A (T/dt)**paths array that holds the simulated
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

class Heston:
    '''
    Simulate the motion of an underlying stock that follows a standard
    Weiner process for T/dt steps over a specified number of paths,
    with stochastic volatility.

    Plot the underlying motion and volatility.

    Determine the pricing of various options using the simulated stochastic
    motion of the underlying asset.

    '''
    def __init__(self,s0,r,T,vol,phi,kappa,xi,dt=0.001,paths=10000):
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

        * numpy matrices will begin to break once a large enough path is chosen
        ** similarly, if too small of a dt is chosen, the numpy matrices
        will begin to break

        Self
        ----
        timeMatrix : numpy.array
            A (T/dt)**paths array that holds all the time values, increasing
            from zero to time T
        S : numpy.array
            A (T/dt)**paths array that holds the simulated
            stock price values
        volMotion : numpy.array
            A (T/dt)**paths array that holds the simulated
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
        [self.S, self.volMotion] = HestonSim(s0,r,vol,phi,kappa,xi,dt,intervals,paths)
        self.simtime = time.time() - start



