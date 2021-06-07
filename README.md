# qcfoptions
Option Calculator and Simulator

Git Repository : [https://github.com/austingriffith94/qcfoptions](https://github.com/austingriffith94/qcfoptions)

An option calculator born from the need to calculate the prices of various options in the QCF program at Georgia Tech. This package provides:

* Black-Scholes pricing of traditional, barrier and exotic options
* Greeks of European style options
* Simulations of underlying asset using stochastic processes
* Pricing of options utilizing the simulated motion of the underlying

This was made initially to avoid rewriting Black-Scholes calculators each time a problem called for it. From there it evolved to incoporate various types of exotic options, as well as a means of simulating the option paths. I'm hoping it can also provide an outlet for those looking for a general code/framework to help in the creation and experimentation of their own option simulations. Each function and class has an explanation on what it does, should the user be interested. For example, if you want to know how to work the European option function, simply type :

    >>> from qcfoptions import bsoptions
    >>> help(bsoptions.EuroOptions)

into the command console, and it should return a relatively complete description of the function and its output.

You can install this package from PyPI by using the command :

    pip install qcfoptions
