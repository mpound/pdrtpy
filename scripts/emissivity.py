#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 08:30:50 2025

@author: mpound
"""
import numpy as np
import matplotlib.pyplot as plt

# emissiviity as function of inclination angle
OI63 = np.array([[90, 6.71e-3], [75, 1.13e-2], [60, 1.17e-2], [45, 1.05e-2], [30, 9.72e-3]])

CII158 = np.array([[90, 1.77e-3], [75, 1.74e-3], [60, 1.19e-3], [45, 8.86e-4], [30, 7.37e-4]])

OI145 = np.array([[90, 2.33e-3], [75, 1.66e-3], [60, 1.01e-3], [45, 7.07e-4], [30, 5.74e-4]])

fig, ax = plt.subplots()
ax.plot(OI63[:, 0], OI63[:, 1], label="OI63", marker=".")
ax.plot(CII158[:, 0], CII158[:, 1], label="CII158", marker=".")
ax.plot(OI145[:, 0], OI145[:, 1], label="OI145", marker=".")
ax.legend()
