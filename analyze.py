#!/usr/bin/env python3


import glob
import os
import string
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
from skimage.morphology import dilation,remove_small_holes, square
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

# colors = list(mcolors.CSS4_COLORS)[10::3]
colors2 = ['orangered','teal', 'firebrick', 'dodgerblue', 'gold', 'forestgreen', 'darkred', 'darkorchid', 'darkorange', 'cornflowerblue', 'darkgreen', 'crimson', 'peru', 'olivedrab']
colors  = ['orangered','teal', 'firebrick', 'dodgerblue', 'gold', 'forestgreen', 'darkred', 'darkorchid', 'darkorange', 'cornflowerblue', 'darkgreen', 'crimson', 'peru', 'olivedrab']
alpha = 1.0
colors_rgb = []
for color in colors:
    c = matplotlib.colors.to_rgba(color)
    colors_rgb.append((c[0],c[1],c[2],alpha))
colors = colors_rgb


#rootdir = '/home/jello/results/AB-block-21-Jun-2022/non-polar'

rootdir = os.getcwd()
dirs = glob.glob(f'{rootdir}/[!_]*/')
dirs.sort()

polymer_rich = []
salt_rich = []

polymer_poor = []
salt_poor = []

CHI_PS = []
HIST_DATA = []
CLUST_SIZE = []


with open(f"{rootdir}/summary.txt", "w") as f:
    f.writelines("chi_PS rho_p_rich rho_p_poor rho_ion_rich rho_ion_poor\n")

with open(f"{rootdir}/summary2.txt", "w") as f:
    f.writelines("chi_PS rho_pol_mean rho_salt_mean std_pol std_salt max_pol max_salt\n")

for dir in dirs:

    hist_data = []

    chi_ps = float(dir[-5:-1])
    dir_name = f"{rootdir}/results"
    os.system(f"mkdir {dir_name}")

    data = []

    files = glob.glob(f'{dir}*.tec')
    files.sort()

    file = files[-1]
    print(f"\n\nAnalyzing: {file}")

    rho1_clusters = []
    rho1_clusters_ = []

    file_name = file.split("/")[-3] + "-" + file.split("/")[-2]

    f = np.loadtxt(file,skiprows = 3)
    xslices = list(set(f[:,0]))
    xslices.sort()

    xdata = f[:,0]
    ydata = f[:,1]
    zdata = f[:,2]

    rhoA = f[:,3]
    rhoB = f[:,4]
    rhoC = f[:,5]

    rhoW = f[:,6]
    rhoCAT = f[:,7]
    rhoANI = f[:,8]
    rhoCI = f[:,9]

    data = [xdata,ydata,zdata,rhoA,rhoB,rhoC,rhoW,rhoCAT,rhoANI,rhoCI]

    fig, ax = plt.subplots(1,3)
    camera = Camera(fig)

    ydata = []
    zdata = []

    rho1data = []
    rho2data = []

    print("Collecting slice data...")

    for xslice in xslices:

        rho1_ = []
        rho2_ = []

        x = data[0]
        # y = data[1]
        # z = data[2]

        rho1 = data[3] + data[4] + data[5] #polymer density
        rho2 = data[9] #counter-ion density

        for i in range(len(rho1)):
            if x[i] == xslice:
                # ydata.append(y[i])
                # zdata.append(z[i])
                rho1_.append(rho1[i])
                rho2_.append(rho2[i])
        rho1data.append(rho1_)
        rho2data.append(rho2_)

        # ydata = np.array(ydata)
        # zdata = np.array(zdata)

    rho1data = np.array(rho1data)
    rho2data = np.array(rho2data)

    # normalize the data

    mean1 = (rho1data.ravel()).mean()
    mean2 = (rho2data.ravel()).mean()

    rho1data /=mean1
    rho2data /=mean2

    m1 = (rho1data.ravel()).mean()
    m2 = (rho2data.ravel()).mean()

    max1 = (rho1data.ravel()).max()
    max2 = (rho2data.ravel()).max()

    std1 = np.std(rho1data.ravel())
    std2 = np.std(rho2data.ravel())

    print(f"Average polymer density: {mean1}, average CI density: {mean2}")
    print(f"Max polymer density: {max1}, max CI density: {max2}")

    with open(f"{rootdir}/summary2.txt", "a+") as f:
        f.writelines(f'{chi_ps} {mean1} {mean2} {std1} {std2} {max1} {max2}\n')

    
    rho1rich = []
    rho2rich = []

    rho1poor = []
    rho2poor = []

    #density correlation

    for x,xslice in enumerate(xslices):
        density_corr = []
        for i in range(len(rho1data[x])):
            density_corr.append(np.exp(-abs((rho1data[x][i] - m1)/std1 - (rho2data[x][i] - m2)/std2 )))
        density_corr = np.array(density_corr)

    #thresholded data - only pick regions that have polymer density beyond one std from the mean

        rho1_thr = []
        for i in range(len(rho1data[x])):
            if abs(rho1data[x][i] - m1) < std1:
                rho1_thr.append(0)
            else:
                rho1_thr.append(1)
        rho1_thr = np.array(rho1_thr)

        zdim, ydim = (rho1_thr.reshape(-1,len(xslices))).shape

        img  = rho1_thr.reshape(-1,len(xslices))
        #img = dilation(img,square(9))
        labels = label(img)
        regions = regionprops(labels)

        for point in rho1data[x]:
            hist_data.append(point)

        # Region preview - sanity check
        # fig, ax = plt.subplots()
        # ax.imshow(img, cmap=plt.cm.gray)

        # for j,props in enumerate(regions):

        #     minr, minc, maxr, maxc = props.bbox
        #     bx = (minc, maxc, maxc, minc, minc)
        #     by = (minr, minr, maxr, maxr, minr)
        #     ax.plot(bx, by, c = colors[j], linewidth=0.5)
        # ax.set_title("Selected clusters - 2D slice")
        # plt.show()

        # Diplay density data

        ax[0].imshow(rho1data[x].reshape(-1,len(xslices)), cmap = "Reds")
        ax[0].set_xlim(0,ydim)
        ax[0].set_ylim(0,zdim)
        ax[0].set_title(r"Polymer density")
        ax[0].text(ydim + 1, 0.0, f"X: {xslice}")
        ax[0].set_xlabel('Y')
        ax[0].set_ylabel('Z')

        ax[1].imshow(density_corr.reshape(-1,len(xslices)), cmap = "Reds")
        ax[1].set_xlim(0,ydim)
        ax[1].set_ylim(0,zdim)
        ax[1].set_title(r"Density correlation")
        ax[1].text(ydim + 1, 0.0, f"X: {xslice}")
        ax[1].set_xlabel('Y')
        ax[1].set_ylabel('Z')

        ax[2].imshow(rho2data[x].reshape(-1,len(xslices)), cmap = "Reds")
        ax[2].set_xlim(0,ydim)
        ax[2].set_ylim(0,zdim)
        ax[2].set_title(r"Ion density")
        ax[2].text(ydim + 1, 0.0, f"X: {xslice}")
        ax[2].set_xlabel('Y')
        ax[2].set_ylabel('Z')

        camera.snap()

        #ax.plot_trisurf(ydata,zdata,rho1data, cmap='viridis', edgecolor='none',alpha = 0.5)
        #ax.plot_surface(ydata[:-1].reshape(2,-1)[:-1].T, zdata[:-1].reshape(2,-1).T, rho1data[:-1].reshape(2,-1).T,cmap = 'viridis')
        #ax.scatter(ydata,zdata,rho1data, cmap='viridis', edgecolor='none',alpha = 0.5)


        # find region-specific concentration

        thr = np.argwhere(rho1_thr == 1)

        for r1,r2 in zip(rho1data[x][thr],rho2data[x][thr]):
            rho1rich.append(r1)
            rho2rich.append(r2)

        thr2 = np.argwhere(rho1_thr == 0)
        for r1,r2 in zip(rho1data[x][thr2],rho2data[x][thr2]):
            rho1poor.append(r1)
            rho2poor.append(r2)


# weights = []
# fig, ax = plt.subplots()
# for i in range(len(HIST_DATA)):
#     weights.append(np.ones(len(HIST_DATA[i]))/len(HIST_DATA[i]))
# ax.hist(HIST_DATA, bins = 30, color=colors2[:len(HIST_DATA)], label=CHI_PS, weights=weights)
# ax.set_xlabel(r"$\frac{\rho}{\rho_{mean}}$")
# ax.set_ylabel("Volume fraction")
# ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
# ax.set_yscale('log')
# plt.suptitle("Polymer clusters")
# plt.tight_layout()
# plt.legend()
# plt.savefig(f"histogram.png", dpi = 500)


clusters = open(f"hist.pkl", "rb")
j = 0
cluster_sizes = []
fig = plt.figure(dpi = 300)
ax = fig.add_subplot(1, 1, 1, projection='3d')
for cluster in clusters:
    if cluster != None:
        x = []
        y = []
        z = []

        for slice in cluster:

            x_ = slice[0]
            y_ = slice[1][:,1]
            z_ = slice[1][:,0]
            
            for i in range(len(y_)):
                x.append(x_)
                y.append(y_[i])
                z.append(z_[i])

        if True:

            voxel = np.zeros((len(xslices),ydim,zdim))
            for i in range(len(x)):
                voxel[x[i],y[i],z[i]] = 1
            color = colors[j]
            ax.voxels(voxel, facecolors = color)

            # ax.scatter(x,z,y,s = 10.0, c = [j] * len(z), cmap = "viridis", alpha = 0.25, label = f"{j}", norm = plt.Normalize(0, 10))
            # ax.scatter(x,y,z,s = 0.2, c = colors[j], alpha = 1.0, label = f"{j}")

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_box_aspect((len(xslices),ydim,zdim))  # aspect ratio is 1:1:1 in data space
            j += 1

plt.title("Polymer density voxels")
plt.show()

#plt.savefig(f"{dir_name}/voxels{str(chi_ps)}.png", dpi = 500)
