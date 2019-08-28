#!/usr/bin/env python3
import numpy as np
from scipy import stats
import math

import os

## Parameters
MAP_SIZE = 201
FLIGHT_RANGE = 31
VELOCITY = 2
TRANSITION_STD = 0.5
STARTING_POS = 20
N = 200
GENERATOR_START = 30
GENERATOR_MIN = 0
GENERATOR_MAX = 60
PRIOR_LOW = 0
PRIOR_HIGH = 200
OBSERVATION_STD = 5

ALTITUDE = 70
PARTICLE_VISUAL_ALTITUDE = 80

# Map generator function
def map_gen(prev, x):
    return math.sin(0.5*x) + stats.norm.rvs(loc=prev, scale=1)

## Set default seed for reproducibility
np.random.seed(0)

## Function for printing data to pgfplots file
def pgf_print(filename, *args):
    f = open(filename, 'w')
    for i in range(args[0].size):
        f.write(str(args[0][i]))
        for arg in args[1:]:
            f.write('\t')
            f.write(str(arg[i]))
        f.write('\n')

## Generate map
a = np.empty(MAP_SIZE)
a[0] = GENERATOR_START
for i in range(1, MAP_SIZE):
    tmp = map_gen(a[i-1], i)
    if tmp < GENERATOR_MIN:
        a[i] = GENERATOR_MIN
    elif tmp > GENERATOR_MAX:
        a[i] = GENERATOR_MAX
    else:
        a[i] = tmp

# Add a flat area in the middle
#a[40:80] = 0

# Print map for use in pgfplots
pgf_print('map.dat', np.arange(MAP_SIZE), a)

# Define the map function
def g_map(x):
    p = math.floor(x)
    n = math.ceil(x)
    if p < 0 or n > MAP_SIZE - 1:
        return 100000
    else:
        return ALTITUDE - (a[p] + (a[n] - a[p]) * (x - p))

# Vectorize g_map
g_map = np.vectorize(g_map)

## Simulate system and generate observations
pos = np.empty(FLIGHT_RANGE)
obs = np.empty(FLIGHT_RANGE)
pos[0] = STARTING_POS
obs[0] = stats.norm.rvs(loc=g_map(pos[0]), scale=OBSERVATION_STD)

def sim_print(i):
    # Print position for use in pgfplots
    pgf_print('pos_' + str(i + 1) + '.dat',
            np.array([pos[i]]),
            np.array([float(ALTITUDE)]))

    # Print vertical distance to ground for use in pgfplots
    pgf_print('dist_' + str(i + 1) + '.dat',
            np.array([pos[i], pos[i]]),
            np.array([float(ALTITUDE), ALTITUDE - g_map(pos[i])]))

    # Print horisontal ground level for use in pgfplots
    pgf_print('level_' + str(i + 1) + '.dat',
            np.array([0, MAP_SIZE - 1]),
            np.array([ALTITUDE - g_map(pos[i]), ALTITUDE - g_map(pos[i])]))

sim_print(0)
for i in range(1, FLIGHT_RANGE):
    pos[i] = stats.norm.rvs(loc=pos[i-1] + VELOCITY, scale=TRANSITION_STD)
    obs[i] = stats.norm.rvs(loc=g_map(pos[i]), scale=OBSERVATION_STD)
    sim_print(i)


## Simulate SIS

# Print data function for pgfplots
def particle_print(s, t, x, w):
    pgf_print(s + '_' + str(t + 1) + '.dat',
            x, np.full(N, float(PARTICLE_VISUAL_ALTITUDE)), w)


# Sample the prior distribution
x = stats.uniform.rvs(loc=PRIOR_LOW, scale=PRIOR_HIGH, size=N)

particle_print('SIS_trans', 0, x, np.full(N, 1 / N))

# Update weights
w = stats.norm.logpdf(obs[0], loc=g_map(x), scale=OBSERVATION_STD)

# Normalize weights
w = w - np.amax(w)
w = np.exp(w) / np.sum(np.exp(w))

particle_print('SIS', 0, x, w)

for t in range(1, FLIGHT_RANGE):

    # Propagate
    x = np.random.normal(loc=x + VELOCITY, scale=TRANSITION_STD)

    particle_print('SIS_trans', t, x, w)

    # Update weights
    w = stats.norm.logpdf(obs[t], loc=g_map(x), scale=OBSERVATION_STD) \
            + np.log(w)

    # Normalize weights
    w = w - np.amax(w)
    w = np.exp(w) / np.sum(np.exp(w))

    particle_print('SIS', t, x, w)


## Simulate BPF

# Sample the prior distribution
x = stats.uniform.rvs(loc=PRIOR_LOW, scale=PRIOR_HIGH, size=N)

particle_print('BPF_trans', 0, x, np.full(N, 1 / N))

# Update weights
w = stats.norm.logpdf(obs[0], loc=g_map(x), scale=OBSERVATION_STD)

# Normalize weights
w = w - np.amax(w)
w = np.exp(w) / np.sum(np.exp(w))

# Print data for pgfplots
particle_print('BPF', 0, x, w)

for t in range(1, FLIGHT_RANGE):

    # Resample
    x = np.random.choice(x, size=N, p=w)
    w = np.full(N, 1 / N)

    # Print resample data for pgfplots
    particle_print('BPF_resample' , t-1, x, w)

    # Propagate
    x = np.random.normal(loc=x + VELOCITY, scale=TRANSITION_STD)

    # Print transition data for pgfplots
    particle_print('BPF_trans', t, x, w)

    # Update weights
    w = stats.norm.logpdf(obs[t], loc=g_map(x), scale=OBSERVATION_STD) \
            + np.log(w)

    # Normalize weights
    w = w - np.amax(w)
    w = np.exp(w) / np.sum(np.exp(w))

    # Print data for pgfplots
    particle_print('BPF', t, x, w)
