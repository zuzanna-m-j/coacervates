#!/usr/bin/env python3

import glob
import os
import numpy as np
from celluloid import Camera
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
import pickle
from matplotlib import cm
import matplotlib.colors as mcolors
import copy
from skimage.morphology import dilation,remove_small_holes, square, remove_small_objects, closing, disk
import matplotlib.colors
import sys
import matplotlib.ticker as mtick
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--chi', default = "chips")
args = parser.parse_args()

if args.chi == "chips":
    chi_name = r"$\chi_{PS}$"
elif args.chi == "chipi":
    chi_name = r"$\chi_{PI}$"

colorsCS = list(mcolors.CSS4_COLORS)[10::5]
colors = ['orangered','teal', 'dodgerblue', 'gold', 'forestgreen', 'darkred', 'darkorchid', 'darkorange', 'cornflowerblue', 'darkgreen', 'crimson', 'peru', 'olivedrab']

rootdir = os.getcwd()
dirs = glob.glob(f'{rootdir}/[!_]*/')
dirs.sort()
with open(f"{rootdir}/data-summary.txt", "w") as f:
    f.writelines("chi_PS rho_p_rich rho_p_poor rho_ion_rich rho_ion_poor\n")

for dir in dirs[-1:]:
    print(dir)

    p_labels = ["xx", "yy", "zz"]

    file = glob.glob(f'{dir}data.dat')[0]
    print(f"\n\nAnalyzing: {file}")

    file_name = "Blank" #file.split("/")[-3] + "-" + file.split("/")[-2]
    data_file = np.loadtxt(file,skiprows = 3)

    time_step = data_file[:,0]

    UPe = data_file[:,1]

    UBond = data_file[:,2]

    Pressure = []
    for i in range(3,9):
        Pressure.append(data_file[:,i])

    UGauss = []
    for i in range(9, 9+28):
        UGauss.append(data_file[:,i])

    UGauss = np.array(UGauss)
    UBond = np.array(UBond)
    UPe = np.array(UPe)
    UGaussSum = np.sum(UGauss, axis = 0)

    UEl = UPe - UGaussSum -UBond

    EN = [UPe, UEl, UGaussSum, UBond]

    en_labels = ["UPe", "UEl", "UGaussSum", "UBon"]

    fig, ax = plt.subplots(2,1)
    fig.dpi = 300

    for i, e in enumerate(EN[:2]):
        ax[0].plot(time_step, e, label = en_labels[i], color = colors[i])

    ax[0].set_xlabel("Time Step")
    ax[0].set_ylabel("Energy")
    #ax[0].set_yscale('log')
    ax[0].legend()

    for i, p in enumerate(Pressure[:3]):
        ax[1].plot(time_step,p, label = p_labels[i], color = colors[i])

    ax[1].set_xlabel("Time Step")
    ax[1].set_ylabel("Pressure")
    ax[1].legend()

    plt.suptitle(f"{file_name}")
    plt.tight_layout()
    plt.show()
    #plt.savefig(f"histogram-clust.png", dpi = 300)

    fig, ax = plt.subplots(2,1)
    fig.dpi = 300

    types = ["A","B","C","W","CAT","ANI","CI","D"]

    ax[0].plot(time_step,UGaussSum, label = 'UGaussSum', color = "dodgerblue")
    ax[0].set_xlabel("Time Step")
    ax[0].set_ylabel("Energy")
    ax[0].legend(loc = 'best')

    g_labels = []
    for i in range(7):
        for j in range(i,7):
            g_labels.append(f"{types[i]} {types[j]}")
    c = 0
    for i,ugass in enumerate(UGauss):
        if ugass[0] != 0 and i not in (18,21):
            ax[1].plot(time_step,ugass, label = g_labels[i], color = colors[c])
            c += 1

    ax[1].set_xlabel("Time Step")
    ax[1].set_ylabel("Energy")
    ax[1].legend(loc = 'best')
    plt.tight_layout()
    plt.show()
