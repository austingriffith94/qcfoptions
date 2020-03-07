'''
Austin Griffith
barriers.py

Provides an analytical solution to barrier options
'''

import numpy as np
from scipy.stats import norm


class BarrierOptions:
    def __init__(self, s, k, r, Z, T, vol, q):
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
        s : number of any type (int, float8, float64 etc.)
            Spot value of underlying asset at current time, t
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation
        r : number of any type (int, float8, float64 etc.)
            Risk free interest rate, implied constant till expiration
        Z : number of any type (int, float8, float64 etc.)
            Barrier value of option, determined at initiation
        T : number of any type (int, float8, float64 etc.)
            Time till expiration for option, can be interpreted as 'T - t' should
            the option already be initiated, and be 't' time from time = 0
        vol : number of any type (int, float8, float64 etc.)
            Volatility of underlying, implied constant till expiration in Black
            Scholes model
        q : number of any type (int, float8, float64 etc.)
            Continuous dividend payout, as a percentage
            
        '''
        
        self.s = s
        self.k = k
        self.r = r
        self.Z = Z
        self.T = T
        self.vol = vol
        self.q = q
        
        self.L = self._lambdaVal(r, vol, q)


    @staticmethod
    def _lambdaVal(r, vol, q):
        '''
        
        Lambda constant calculated from the risk-netural
        underlying drift.

        Parameters
        ----------
        r : number of any type (int, float8, float64 etc.)
            Risk free interest rate, implied constant till expiration
        vol : number of any type (int, float8, float64 etc.)
            Volatility of underlying, implied constant till expiration in Black
            Scholes model
        q : number of any type (int, float8, float64 etc.)
            Continuous dividend payout, as a percentage

        Returns
        -------
        l : float
            The lambda constant used in each of analytical solutions

        '''
        
        mu = (r - q - vol*vol*0.5)
        l = 1 + (mu / (vol*vol))
        return(l)


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
        
        xval = np.log(self.s / self.k) / (self.vol*np.sqrt(self.T)) + self.L*self.vol*np.sqrt(self.T)
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
        
        xval = np.log(self.s / self.Z) / (self.vol*np.sqrt(self.T)) + self.L*self.vol*np.sqrt(self.T)
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
        
        xval = np.log(np.square(self.Z) / (self.s*self.k)) / (self.vol*np.sqrt(self.T)) + self.L*self.vol*np.sqrt(self.T)
        partial = alpha*self.s*np.power(self.Z / self.s, 2*self.L)*norm.cdf(beta*xval) - \
            alpha*self.k*np.exp(-1*self.r*self.T)*np.power(self.Z / self.s, 2*self.L - 2)*norm.cdf(beta*xval - beta*self.vol*np.sqrt(self.T))
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
        
        xval = np.log(self.Z / self.s) / (self.vol*np.sqrt(self.T)) + self.L*self.vol*np.sqrt(self.T)
        partial = alpha*self.s*np.power(self.Z / self.s, 2*self.L)*norm.cdf(beta*xval) - \
            alpha*self.k*np.exp(-1*self.r*self.T)*np.power(self.Z / self.s, 2*self.L - 2)*norm.cdf(beta*xval - beta*self.vol*np.sqrt(self.T))
        return(partial)



    def DownOutPutLow(self):
        '''
        
        Calculate the Down-and-Out PUT option, 
        where the barrier is LESS THAN the strike*
    
        Parameters
        ----------
        s : number of any type (int, float8, float64 etc.)
            Spot value of underlying asset at current time, t
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation
        r : number of any type (int, float8, float64 etc.)
            Risk free interest rate, implied constant till expiration
        Z : number of any type (int, float8, float64 etc.)
            Barrier value of option, determined at initiation
        T : number of any type (int, float8, float64 etc.)
            Time till expiration for option, can be interpreted as 'T - t' should
            the option already be initiated, and be 't' time from time = 0
        vol : number of any type (int, float8, float64 etc.)
            Volatility of underlying, implied constant till expiration in Black
            Scholes model
        q : number of any type (int, float8, float64 etc.)
            Continuous dividend payout, as a percentage
    
        Notes
        -----
        * Z < k must hold true for this function
    
        Returns
        -------
        price : float
            Price value of barrier option. 
    
        '''
        
        a = -1
        b = 1
        
        if self.Z >= self.k:
            return None
    
        if self.s >= self.Z:
            price = self._I1(a,b) - self._I2(a,b) + self._I3(a,b) - self._I4(a,b)
        else:
            price = 0.0
        return(max(price, 0.0))
    
    
    def DownOutCall(self):
        '''
        
        Calculate the Down-and-Out CALL option, for any barrier
    
        Parameters
        ----------
        s : number of any type (int, float8, float64 etc.)
            Spot value of underlying asset at current time, t
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation
        r : number of any type (int, float8, float64 etc.)
            Risk free interest rate, implied constant till expiration
        Z : number of any type (int, float8, float64 etc.)
            Barrier value of option, determined at initiation
        T : number of any type (int, float8, float64 etc.)
            Time till expiration for option, can be interpreted as 'T - t' should
            the option already be initiated, and be 't' time from time = 0
        vol : number of any type (int, float8, float64 etc.)
            Volatility of underlying, implied constant till expiration in Black
            Scholes model
        q : number of any type (int, float8, float64 etc.)
            Continuous dividend payout, as a percentage
    
        Returns
        -------
        price : float
            Price value of barrier option.
    
        '''
        
        a = 1
        b = 1
    
        if self.k > self.Z and self.s >= self.Z:
            price = self._I1(a,b) - self._I3(a,b)
        elif self.k <= self.Z and self.s >= self.Z:
            price = self._I2(a,b) - self._I4(a,b)
        else:
            price = 0.0
        return(max(price, 0.0))
    
    
    def UpOutCallHigh(self):
        '''
        
        Calculate the Up-and-Out CALL option, 
        where the barrier is GREATER THAN the strike*
    
        Parameters
        ----------
        s : number of any type (int, float8, float64 etc.)
            Spot value of underlying asset at current time, t
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation
        r : number of any type (int, float8, float64 etc.)
            Risk free interest rate, implied constant till expiration
        Z : number of any type (int, float8, float64 etc.)
            Barrier value of option, determined at initiation
        T : number of any type (int, float8, float64 etc.)
            Time till expiration for option, can be interpreted as 'T - t' should
            the option already be initiated, and be 't' time from time = 0
        vol : number of any type (int, float8, float64 etc.)
            Volatility of underlying, implied constant till expiration in Black
            Scholes model
        q : number of any type (int, float8, float64 etc.)
            Continuous dividend payout, as a percentage
    
        Notes
        -----
        * Z > k must hold true for this function
    
        Returns
        -------
        price : float
            Price value of barrier option.
    
        '''
        
        a = 1
        b = -1
        
        if self.Z <= self.k:
            return None
    
        if self.s <= self.Z:
            price = self._I1(a,b) - self._I2(a,b) + self._I3(a,b) - self._I4(a,b)
        else:
            price = 0.0
        return(max(price, 0.0))
    
    
    def UpOutPut(self):
        '''
        
        Calculate the Up-and-Out PUT option, for any barrier
    
        Parameters
        ----------
        s : number of any type (int, float8, float64 etc.)
            Spot value of underlying asset at current time, t
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation
        r : number of any type (int, float8, float64 etc.)
            Risk free interest rate, implied constant till expiration
        Z : number of any type (int, float8, float64 etc.)
            Barrier value of option, determined at initiation
        T : number of any type (int, float8, float64 etc.)
            Time till expiration for option, can be interpreted as 'T - t' should
            the option already be initiated, and be 't' time from time = 0
        vol : number of any type (int, float8, float64 etc.)
            Volatility of underlying, implied constant till expiration in Black
            Scholes model
        q : number of any type (int, float8, float64 etc.)
            Continuous dividend payout, as a percentage
    
        Returns
        -------
        price : float
            Price value of barrier option.

        '''
        
        a = -1
        b = -1
    
        if self.k <= self.Z and self.s <= self.Z:
            price = self._I1(a,b) - self._I3(a,b)
        elif self.k > self.Z and self.s <= self.Z:
            price = self._I2(a,b) - self._I4(a,b)
        else:
            price = 0.0
        return(max(price, 0.0))
    
    
    def DownInCall(self):
        '''
        
        Calculate the Down-and-In CALL option, for any barrier
    
        Parameters
        ----------
        s : number of any type (int, float8, float64 etc.)
            Spot value of underlying asset at current time, t
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation
        r : number of any type (int, float8, float64 etc.)
            Risk free interest rate, implied constant till expiration
        Z : number of any type (int, float8, float64 etc.)
            Barrier value of option, determined at initiation
        T : number of any type (int, float8, float64 etc.)
            Time till expiration for option, can be interpreted as 'T - t' should
            the option already be initiated, and be 't' time from time = 0
        vol : number of any type (int, float8, float64 etc.)
            Volatility of underlying, implied constant till expiration in Black
            Scholes model
        q : number of any type (int, float8, float64 etc.)
            Continuous dividend payout, as a percentage
    
        Returns
        -------
        price : float
            Price value of barrier option.

        '''
        
        a = 1
        b = 1
    
        if self.k > self.Z:
            price = self._I3(a,b)
        elif self.k <= self.Z:
            price = self._I1(a,b) - self._I2(a,b) + self._I4(a,b)
        else:
            price = 0.0
        return(max(price, 0.0))
    
    
    def DownInPut(self):
        '''
        
        Calculate the Down-and-In PUT option, for any barrier
    
        Parameters
        ----------
        s : number of any type (int, float8, float64 etc.)
            Spot value of underlying asset at current time, t
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation
        r : number of any type (int, float8, float64 etc.)
            Risk free interest rate, implied constant till expiration
        Z : number of any type (int, float8, float64 etc.)
            Barrier value of option, determined at initiation
        T : number of any type (int, float8, float64 etc.)
            Time till expiration for option, can be interpreted as 'T - t' should
            the option already be initiated, and be 't' time from time = 0
        vol : number of any type (int, float8, float64 etc.)
            Volatility of underlying, implied constant till expiration in Black
            Scholes model
        q : number of any type (int, float8, float64 etc.)
            Continuous dividend payout, as a percentage
    
        Returns
        -------
        price : float
            Price value of barrier option.
    
        '''
        
        a = -1
        b = 1
    
        if self.k > self.Z:
            price = self._I2(a,b) - self._I3(a,b) + self._I4(a,b)
        elif self.k <= self.Z:
            price = self._I1(a,b)
        else:
            price = 0.0
        return(max(price, 0.0))
    
    
    def UpInCall(self):
        '''
        Calculate the Up-and-In Call option, for any barrier
    
        Parameters
        ----------
        s : number of any type (int, float8, float64 etc.)
            Spot value of underlying asset at current time, t
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation
        r : number of any type (int, float8, float64 etc.)
            Risk free interest rate, implied constant till expiration
        Z : number of any type (int, float8, float64 etc.)
            Barrier value of option, determined at initiation
        T : number of any type (int, float8, float64 etc.)
            Time till expiration for option, can be interpreted as 'T - t' should
            the option already be initiated, and be 't' time from time = 0
        vol : number of any type (int, float8, float64 etc.)
            Volatility of underlying, implied constant till expiration in Black
            Scholes model
        q : number of any type (int, float8, float64 etc.)
            Continuous dividend payout, as a percentage
    
        Returns
        -------
        price : float
            Price value of barrier option.
    
        '''
        
        a = 1
        b = -1
    
        if self.k > self.Z:
            price = self._I1(a,b)
        elif self.k <= self.Z:
            price = self._I2(a,b) - self._I3(a,b) + self._I4(a,b)
        else:
            price = 0.0
        return(max(price, 0.0))
    
    
    def UpInPut(self):
        '''
        
        Calculate the Up-and-In Put option, for any barrier
    
        Parameters
        ----------
        s : number of any type (int, float8, float64 etc.)
            Spot value of underlying asset at current time, t
        k : number of any type (int, float8, float64 etc.)
            Strike value of option, determined at initiation
        r : number of any type (int, float8, float64 etc.)
            Risk free interest rate, implied constant till expiration
        Z : number of any type (int, float8, float64 etc.)
            Barrier value of option, determined at initiation
        T : number of any type (int, float8, float64 etc.)
            Time till expiration for option, can be interpreted as 'T - t' should
            the option already be initiated, and be 't' time from time = 0
        vol : number of any type (int, float8, float64 etc.)
            Volatility of underlying, implied constant till expiration in Black
            Scholes model
        q : number of any type (int, float8, float64 etc.)
            Continuous dividend payout, as a percentage
    
        Returns
        -------
        price : float
            Price value of barrier option.
    
        '''
        
        a = -1
        b = -1
    
        if self.k > self.Z:
            price = self._I1(a,b) - self._I2(a,b) + self._I4(a,b)
        elif self.k <= self.Z:
            price = self._I3(a,b)
        else:
            price = 0.0
        return(max(price, 0.0))
