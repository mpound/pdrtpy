#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 10:56:13 2025

@author: mpound
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


# partition funciton
def zt(temp, file, clip):
    # sum gu*exp(-Eu/T)
    table = Table.read(file, format="ipac")
    if clip:
        idc = np.where(table["Tu"] < 25000)
        gu = table["gu"][idc]
        eu = table["Tu"][idc]
    else:
        gu = table["gu"]
        eu = table["Tu"]
    return np.array([np.sum(gu * np.exp(-eu / t)) for t in temp])


def z(temp):
    return 0.0247 * temp / (1.0 - np.exp(-6000.0 / temp))


def plot(temp, file):
    fig, ax = plt.subplots()
    # temp = np.arange(10, 10000, 50)
    ax.plot(temp, z(temp), label="Herbst+", color="blue")
    ax.plot(temp, zt(temp, file, True), label="Full (clipped)", color="red", linestyle="--")
    ax.plot(temp, zt(temp, file, False), label="Full", color="green")
    ax.legend()
    plt.show()


"""
excitationfit -> BaseExcitationFit
    ExcitatioFit(molecule: str)
        supported types are H2, CO, 13CO, CH+, [H2O]

        table
        opr
        partition function
"""
