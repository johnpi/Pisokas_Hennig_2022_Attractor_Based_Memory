import numpy as np

# Functions to use for regression to data
func_line_str = 'a*x + b'
def func_line(x, a, b):
    return a * x + b

func_exp1_str='exp(b*x)'
def func_exp1(x, b):
    return np.exp(b*x)

func_exp2_str='exp(b*x) + c'
def func_exp2(x, b, c):
    return np.exp(b*x) + c

func_exp3_str='a * exp(b*x) + c'
def func_exp3(x, a, b, c):
    return a * np.exp(b*x) + c
