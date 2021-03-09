#! /usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import sys
import os
import math

matplotlib.rcParams.update({'font.size': 10})


parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='Input file')
parser.add_argument('--output', required=True, help='Output file')
parser.add_argument('--colorbar', action='store_true', help='Plot color bar')
parser.add_argument('--title', default='a', help='Title string')

args = parser.parse_args()

df = pd.read_csv(args.input, sep=';', header=None)

if len(df.columns) == 4:
    opt = False
elif len(df.columns) == 5:
    opt = True
else:
    raise ValueError('Unexpected format!')

normalized = df[3].values.copy()

min_value = normalized.min()
max_value = normalized.max()

plt.rcParams['axes.facecolor'] = '#a8a8a8'

cmap = LinearSegmentedColormap.from_list("mycmap", ((0.0, 'b'), (0.5, 'w'), (1.0, 'r'),), N=8192)

def normalize(v):
    if v < 0.0:
        return -(v - min_value) / (2 * min_value)
    elif v == 0.0:
        return 0.5
    else:
        return v / (2 * max_value) + 0.5

radius = math.sqrt(2)
delta = 0.05

plt.rc('font', family='serif')

fig = plt.figure(figsize=(3.33, 3.33), dpi=600)

my_circle = plt.Circle((0,0), radius - delta, color=plt.rcParams['axes.facecolor'])
if opt:
    for _, (angle, _, _, v, _) in df.iterrows():
        x = (radius + delta) * math.cos(angle)
        y = (radius + delta) * math.sin(angle)
        if v < 0.0:
            zorder = -500
        elif v > 0.0:
            zorder = -498
        else:
            zorder = -499
        plt.plot((0, x), (0, y), c=cmap(normalize(v)), zorder=zorder, linewidth=1.0)
else:
    for _, (angle, _, _, v) in df.iterrows():
        x = (radius + delta) * math.cos(angle)
        y = (radius + delta) * math.sin(angle)
        if v < 0.0:
            zorder = -500
        elif v > 0.0:
            zorder = -498
        else:
            zorder = -499
        plt.plot((0, x), (0, y), c=cmap(normalize(v)), zorder=zorder, linewidth=1.0)

# this hack is needed so we can use the color map. it has zorder=-500, so will be hidden by
# the circle
plt.scatter((0, 0, 0), (0, 0, 0), c=(0, 0.5, 1.0), cmap=cmap, zorder=-500)

plt.gca().add_artist(my_circle)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'({args.title})', loc='left')

if args.colorbar:
    cbar = plt.colorbar(ticks=(0.0, 0.5, 1.0))
    labels = [f"{v:.1f}" for v in (min_value, 0, max_value)]
    cbar.ax.set_yticklabels(labels)

plt.grid(color='grey')
plt.savefig(args.output, bbox_inches='tight')
