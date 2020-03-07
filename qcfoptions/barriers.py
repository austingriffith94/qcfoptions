'''
Austin Griffith
barriers.py

Provides an analytical solution to barrier options
'''

import numpy as np
from scipy.stats import norm


class BarrierOptions:
    '''
        
    Provides an analytical solution to barrier options using
    the Black Scholes methodology. Due to the underlying calculus'
    assumptions, the risk-free rate and implied volatility of the 
    underlying are held constant from t0 to t1.
    
    f(u) : the density function of the natural logarithm of
        the risk-neutral underlying asset return
    g(u) : the density function of the natural logarithm of
        the risk-neutral underlying asset return when the
        underlying asset price starts above/below the barrier
        crosses the barrier but ends up above/below the barrier
        at expiration
    
    The _I functions are set values used as the analytical 
    solution to the barrier options. Since a barrier option
    is marked by a set of logic conditions being met, these
    can be manipulated via an alpha and beta scalar to account
    for each barrier option's requirements. Each barrier is
    a linear combination of this set of analytical solutions.
    
    I1 - I2 : call payoff integrated over f(u) between the 
        strike and barrier
    I1 - I3 : call payoff integrated over the probability density
        of the terminal asset price conditional on NOT crossing
        the barrier
    I2 - I4 : 
    I3 : call payoff integrated over g(u) conditional on crossing
        the barrier
    I4 : call payoff integrated over the density function of
        the natural logarithm under risk-neutral assumptions 
        between the barrier and infinity
    I5 : rebate for "In" options
    I6 : reabte for "Out" options
    
    
    Parameters
    ----------
    spot : number of any type (int, float8, float64 etc.)
        Spot value of underlying asset at current time, t
    strike : number of any type (int, float8, float64 etc.)
        Strike value of option, determined at initiation
    riskfree : number of any type (int, float8, float64 etc.)
        Risk free interest rate, implied constant till expiration
    barrier : number of any type (int, float8, float64 etc.)
        Barrier value of option, determined at initiation
    tau : number of any type (int, float8, float64 etc.)
        Time till expiration for option, can be interpreted as 'T - t' should
        the option already be initiated, and be 't' time from time = 0
    vol : number of any type (int, float8, float64 etc.)
        Volatility of underlying, implied constant till expiration in Black
        Scholes model
    div : number of any type (int, float8, float64 etc.)
        Continuous dividend payout, as a percentage
    rebate : number of any type (int, float8, float64 etc.)
        Rebate of barrier option, if there is no rebate provision, set = 0
        Default value is 0
        
    '''
    
    def __init__(self, spot, strike, riskfree, barrier, tau, vol, div, rebate=0):
        self.s = spot
        self.k = strike
        self.r = riskfree
        self.Z = barrier
        self.T = tau
        self.vol = vol
        self.q = div
        self.R = rebate
        
        self.L = self._lambdaDrift()


    def _lambdaDrift(self):
        '''
        
        Lambda constant calculated from the risk-netural
        drift of the underlying

        Returns
        -------
        l : float
            The lambda constant used in each of barrier option
            analytical solutions

        '''
        
        m = self._mu()
        l = 1 + (m / (self.vol*self.vol))
        return(l)
    
    
    def _mu(self):
        '''
        
        The underlying's risk-netural drift term,
        referred to as 'Mu'

        Returns
        -------
        mu : float
            The drift term used in the barrier option
            analytical solutions

        '''

        mu = (self.r - self.q - self.vol*self.vol*0.5)
        return(mu)


    def _x1val(self):
        x = np.log(self.s / self.Z) / (self.vol*np.sqrt(self.T)) + self.L*self.vol*np.sqrt(self.T)
        return(x)
    
    def _xval(self):
        x = np.log(self.s / self.k) / (self.vol*np.sqrt(self.T)) + self.L*self.vol*np.sqrt(self.T)
        return(x)
    
    def _y1val(self):
        y = np.log(self.Z / self.s) / (self.vol*np.sqrt(self.T)) + self.L*self.vol*np.sqrt(self.T)
        return(y)
    
    def _yval(self):
        y = np.log(np.square(self.Z) / (self.s*self.k)) / (self.vol*np.sqrt(self.T)) + self.L*self.vol*np.sqrt(self.T)
        return(y)
    
    def _zval(self):
        z = np.log(self.Z / self.s) / (self.vol*np.sqrt(self.T)) + self._bval()*self.vol*np.sqrt(self.T)
        return(z)

    def _aval(self):
        a = self._mu() / (self.vol*self.vol)
        return(a)
    
    def _bval(self):
        b = np.sqrt(self._mu()**2 + 2*self.r*self.vol*self.vol) / (self.vol*self.vol)
        return(b)


    def _I1(self, alpha : int, beta : int):
        '''
        
        Parameters
        ----------
        alpha : int
            Scalar to represent either a call or put
        beta : int
            Scalar to represent whether the asset price
            starts above or below the barrier

        Returns
        -------
        partial : float
            The I1 partial analytical solution for the barrier option
            
        '''
        
        xval = self._xval()
        partial = alpha*self.s*norm.cdf(alpha*xval) - alpha*self.k*np.exp(-1*self.r*self.T)*norm.cdf(alpha*xval - alpha*self.vol*np.sqrt(self.T))
        return(partial)


    def _I2(self, alpha : int, beta : int):
        '''
        
        Parameters
        ----------
        alpha : int
            Scalar to represent either a call or put
        beta : int
            Scalar to represent whether the asset price
            starts above or below the barrier

        Returns
        -------
        partial : float
            The I2 partial analytical solution for the barrier option
            
        '''
        
        xval = self._x1val()
        partial = alpha*self.s*norm.cdf(alpha*xval) - alpha*self.k*np.exp(-1*self.r*self.T)*norm.cdf(alpha*xval - alpha*self.vol*np.sqrt(self.T))
        return(partial)
    
    
    def _I3(self, alpha : int, beta : int):
        '''
        
        Parameters
        ----------
        alpha : int
            Scalar to represent either a call or put
        beta : int
            Scalar to represent whether the asset price
            starts above or below the barrier

        Returns
        -------
        partial : float
            The I3 partial analytical solution for the barrier option
            
        '''
        
        yval = self._yval()
        partial = alpha*self.s*np.power(self.Z / self.s, 2*self.L)*norm.cdf(beta*yval) - \
            alpha*self.k*np.exp(-1*self.r*self.T)*np.power(self.Z / self.s, 2*self.L - 2)*norm.cdf(beta*yval - beta*self.vol*np.sqrt(self.T))
        return(partial)
    
    
    def _I4(self, alpha : int, beta : int):
        '''

        Parameters
        ----------
        alpha : int
            Scalar to represent either a call or put
        beta : int
            Scalar to represent whether the asset price
            starts above or below the barrier

        Returns
        -------
        partial : float
            The I4 partial analytical solution for the barrier option
            
        '''
        
        yval = self._y1val()
        partial = alpha*self.s*np.power(self.Z / self.s, 2*self.L)*norm.cdf(beta*yval) - \
            alpha*self.k*np.exp(-1*self.r*self.T)*np.power(self.Z / self.s, 2*self.L - 2)*norm.cdf(beta*yval - beta*self.vol*np.sqrt(self.T))
        return(partial)
    
    
    def _I5(self, beta : int):
        x = self._x1val()
        y = self._y1val()
        partial = self.R*np.exp(-1*self.r*self.T) * \
            (norm.cdf(beta*x - beta*self.vol*np.sqrt(self.T)) - \
             np.power(self.Z / self.s, 2*self.L - 2)*norm.cdf(beta*y - beta*self.vol*np.sqrt(self.T)))
        return(partial)
    
    
    def _I6(self, beta : int):
        a = self._aval()
        b = self._bval()
        z = self._zval()
        partial = self.R * (np.power(self.Z / self.s, a - b)*norm.cdf(beta*z) - \
                            np.power(self.Z / self.s, a - b)*norm.cdf(beta*z - 2*beta*b*self.vol*np.sqrt(self.T)))
        return(partial)



    def DownOutPut(self):
        '''
        
        Calculate the Down-and-Out PUT option
    
        Returns
        -------
        price : float
            Price value of barrier option. 
    
        '''
        
        a = -1
        b = 1
        
        if self.k > self.Z and self.s >= self.Z:
            price = self._I1(a,b) - self._I2(a,b) + self._I3(a,b) - self._I4(a,b) + self._I6(b)
        elif self.k < self.Z and self.s >= self.Z:
            price = self._I6(b)
        else:
            price = 0.0
        return(max(price, 0.0))
    
    
    def DownOutCall(self):
        '''
        
        Calculate the Down-and-Out CALL option, for any barrier
    
        Returns
        -------
        price : float
            Price value of barrier option.
    
        '''
        
        a = 1
        b = 1
    
        if self.k > self.Z and self.s >= self.Z:
            price = self._I1(a,b) - self._I3(a,b) + self._I6(b)
        elif self.k < self.Z and self.s >= self.Z:
            price = self._I2(a,b) - self._I4(a,b) + self._I6(b)
        else:
            price = 0.0
        return(max(price, 0.0))
    
    
    def UpOutCall(self):
        '''
        
        Calculate the Up-and-Out CALL option
    
        Returns
        -------
        price : float
            Price value of barrier option.
    
        '''
        
        a = 1
        b = -1
        
        if self.k > self.Z and self.s <= self.Z:
            price = self._I1(a,b) - self._I2(a,b) + self._I3(a,b) - self._I4(a,b) + self._I6(b)
        elif self.k < self.Z and self.s <= self.Z:
            price = self._I6(b)
        else:
            price = 0.0
        return(max(price, 0.0))
    
    
    def UpOutPut(self):
        '''
        
        Calculate the Up-and-Out PUT option
    
        Returns
        -------
        price : float
            Price value of barrier option.

        '''
        
        a = -1
        b = -1
    
        if self.k < self.Z and self.s <= self.Z:
            price = self._I1(a,b) - self._I3(a,b) + self._I6(b)
        elif self.k > self.Z and self.s <= self.Z:
            price = self._I2(a,b) - self._I4(a,b) + self._I6(b)
        else:
            price = 0.0
        return(max(price, 0.0))
    
    
    def DownInCall(self):
        '''
        
        Calculate the Down-and-In CALL option
    
        Returns
        -------
        price : float
            Price value of barrier option.

        '''
        
        a = 1
        b = 1
    
        if self.k > self.Z:
            price = self._I3(a,b) + self._I5(b)
        elif self.k < self.Z:
            price = self._I1(a,b) - self._I2(a,b) + self._I4(a,b) + self._I5(b)
        else:
            price = 0.0
        return(max(price, 0.0))
    
    
    def DownInPut(self):
        '''
        
        Calculate the Down-and-In PUT option
    
        Returns
        -------
        price : float
            Price value of barrier option.
    
        '''
        
        a = -1
        b = 1
    
        if self.k > self.Z:
            price = self._I2(a,b) - self._I3(a,b) + self._I4(a,b) + self._I5(b)
        elif self.k < self.Z:
            price = self._I1(a,b) + self._I5(b)
        else:
            price = 0.0
        return(max(price, 0.0))
    
    
    def UpInCall(self):
        '''
        
        Calculate the Up-and-In Call option
    
        Returns
        -------
        price : float
            Price value of barrier option.
    
        '''
        
        a = 1
        b = -1
    
        if self.k > self.Z:
            price = self._I1(a,b) + self._I5(b)
        elif self.k < self.Z:
            price = self._I2(a,b) - self._I3(a,b) + self._I4(a,b) + self._I5(b)
        else:
            price = 0.0
        return(max(price, 0.0))
    
    
    def UpInPut(self):
        '''
        
        Calculate the Up-and-In Put option
    
        Returns
        -------
        price : float
            Price value of barrier option.
    
        '''
        
        a = -1
        b = -1
    
        if self.k > self.Z:
            price = self._I1(a,b) - self._I2(a,b) + self._I4(a,b) + self._I5(b)
        elif self.k < self.Z:
            price = self._I3(a,b) + self._I5(b)
        else:
            price = 0.0
        return(max(price, 0.0))




if __name__ == '__main__':
    opt1 = [1, 2, 0.02, 1.5, 5, 0.05, 0.01, 0.01]
    bro = BarrierOptions(*opt1)
    a = bro.DownInCall()