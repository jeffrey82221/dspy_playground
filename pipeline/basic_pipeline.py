from functools import reduce, partial
from typing import Callable

class Forked(list):
    """ Contains a list of data after forking """

class Fork(list):
    """ Contains a list of functions for forking """

class Reducer(object):
    """ Contains a function for reducing forked data """
    def __init__(self, func):
        self.func = func

def fork(*funcs):
    return Fork(funcs)

def reducer(func):
    """ Return a reducer form based on a function that accepts a
        Forked list as its first argument """
    return Reducer(func)   

def apply_func(data, func):
    """ Apply a function to data which may be forked """
    if isinstance(data, Forked):
        return Forked(apply_func(datum, func) for datum in data)
    else:
        return func(data)

def apply_form(data, form):
    """ Apply a pipeline form (which may be a function, fork, or reducer) 
        to the data """
    if callable(form):
        return apply_func(data, form)
    elif isinstance(form, Fork):
        return Forked(apply_func(data, func) for func in form)
    elif isinstance(form, Reducer):
        return form.func(data)

def pipe(data, *forms):
    """ Apply a pipeline of function forms to data """
    return reduce(apply_form, forms, data)


def double(x): return x * 2
def inc(x): return x + 1
def dec(x): return x - 1
def mult(L): return L[0] * L[1]
def triple(x): return x * 3

print(pipe(10, inc, double)) # 10 + 1 = 11 , 11 * 2 = 22
print(pipe(10, fork(dec, inc), double))  # 10-1 = 9 / 10+1 = 22 => * 2 => [18, 22]
print(pipe(10, fork(dec, inc), double, reducer(mult)))   # 396

def compose(*functions) -> Callable:
    """Composes functions into a single function"""
    return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)

def fork(*functions) -> Callable:
    """Apply multiple functions on same input to diliver multiple outputs"""
    return lambda x: [func(x) for func in functions]

print('compose(double, inc)(2):', 
      compose(double, inc)(2))

print('''
    compose(
      fork(inc, dec), 
      partial(map, double), 
      partial(reduce, lambda x, y: x + y)
    )(2):''', 
    compose(fork(inc, dec), 
            partial(map, double), 
            partial(reduce, lambda x, y: x + y)
    )(2))

