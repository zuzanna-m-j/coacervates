#!/usr/bin/env python3


import glob
import os
from turtle import shape
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
dirs = glob.glob(f'{rootdir}/*/')
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

for dir in dirs:

    hist_data = []

    chi_ps = float(dir[-5:-1])
    dir_name = f"{dir}results"
    print(dir_name)
    os.system(f"mkdir {dir_name}")

    data = []

    files = glob.glob(f'{dir}*.tec')
    files.sort()

    file = files[-1]
    print(f"Analyzing: {file}")

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

    max1 = (rho1data.ravel()).max()
    max2 = (rho2data.ravel()).max()

    rho1data /=max1
    rho2data /=max2

    m1 = (rho1data.ravel()).mean()
    m2 = (rho2data.ravel()).mean()

    std1 = np.std(rho1data.ravel())
    std2 = np.std(rho2data.ravel())

    print("Generating density plots...")

    rho1rich = []
    rho2rich = []

    rho1poor = []
    rho2poor = []

    #density correlation

    for x,xslice in enumerate(xslices):
        density_corr = []
        for i in range(len(rho1data[x])):
            density_corr.append(np.exp(-abs(rho1data[x][i] -  rho2data[x][i])))
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
        img = dilation(img,square(9))
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

        # find clusters
        
    print("Making clusters...") 
    for x,xslice in enumerate(xslices):

        if xslice == xslices[0]:
            for region in regions:
                rho1_clusters.append([[x,region.coords]])

        else:
            for regid,region in enumerate(regions):
                overlap = False
                neighs = []
                coords = region.coords

                for rid, p_region in enumerate(rho1_clusters):
                    if p_region != None:
                        for m in range(len(p_region)):
                            if overlap == True:
                                break
                            dx = abs(x - p_region[m][0])
                            # if dx > len(xslices)/2:
                            #     dx = len(xslices) - dx
                            if dx == 1 or dx == 44:
                                for coord in coords:
                                    if overlap == True:
                                        break
                                    dy  = abs(coord[1] - p_region[m][1][:,1])
                                    for i in range(len(dy)):
                                        if dy[i] >= ydim//2:
                                            dy[i] = ydim - dy[i]

                                    dz  = abs(coord[0] - p_region[m][1][:,0])
                                    for i in range(len(dz)):
                                        if dz[i] >= int(zdim/2):
                                            dz[i] = zdim - dz[i]

                                    dr = dy + dz
                                    if min(dr) <= 1:
                                        overlap = True
                            if overlap == True:
                                neighs.append(rid)
                                overlap = False
                            else:
                                pass
                    
                if len(neighs) != 0:
                    i = neighs[0]
                    for j in neighs:
                        if i != j:
                            cluster = rho1_clusters[j]
                            for slice in cluster:
                                rho1_clusters[i].append(slice)

                    rho1_clusters[i].append([x, region.coords])
                    for j in neighs:
                        if i != j:
                            rho1_clusters[j] = None
                else:
                        rho1_clusters.append([[x, region.coords]])


    HIST_DATA.append(hist_data)
    CHI_PS.append(chi_ps)

    res = camera.animate(interval=500)
    res.save(f'{dir_name}/rho.gif', dpi = 500)

    # plt.show()

    rho1rich = np.array(rho1rich)
    rho1rich_avg  = np.average(rho1rich)

    rho2rich = np.array(rho2rich)
    rho2rich_avg  = np.average(rho2rich)

    rho1poor = np.array(rho1poor)
    rho1poor_avg  = np.average(rho1poor)

    rho2poor = np.array(rho2poor)
    rho2poor_avg  = np.average(rho2poor)

    with open(f"{rootdir}/summary.txt", "a+") as f:
        f.writelines(f'{chi_ps} {(rho1rich_avg-m1)/std1} {(rho1poor_avg-m1)/std1} {(rho2rich_avg - m2)/std2} {(rho2poor_avg - m2)/std2}\n')

    polymer_rich.append((rho1rich_avg-m1)/std1)
    salt_rich.append((rho2rich_avg - m2)/std2)

    polymer_poor.append((rho1poor_avg-m1)/std1)
    salt_poor.append((rho2poor_avg - m2)/std2)
    
    rho_file = open(f"{dir_name}/rho1_clusters.pkl", "wb")
    pickle.dump(rho1_clusters,rho_file)
    rho_file.close()

    rho_file = open(f"{dir_name}/rho1_clusters.pkl", "rb")
    clusters = pickle.load(rho_file)
    rho_file.close()


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

                CLUST_SIZE.append(len(z))

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
    plt.savefig(f"{dir_name}/voxels.png", dpi = 500)


fig, ax = plt.subplots(2,1)
plt.suptitle("Concentration in different phases")

ax[0].set_title("Polymer concentration")
ax[0].plot(CHI_PS, polymer_rich, 'o-', label = "polymer-rich phase")
ax[0].plot(CHI_PS, polymer_poor, 'o-', label = "polymer-poor phase")
ax[0].set_xlabel(r"$\chi_{PS}$")
ax[0].set_ylabel(r"$\frac{\rho_{pol} - <\rho_{pol}>}{\sigma_{pol}}$")
ax[0].legend()

ax[1].set_title("Salt concentration")
ax[1].plot(CHI_PS, salt_rich, 'o-', label = "polymer-rich phase")
ax[1].plot(CHI_PS, salt_rich, 'o-', label = "polymer-rich phase")
ax[1].set_xlabel(r"$\chi_{PS}$")
ax[1].set_ylabel(r"$\frac{\rho_{salt} - <\rho_{salt}>}{\sigma_{salt}}$")
ax[1].legend()

plt.tight_layout()
plt.savefig(fname = f"{rootdir}/density.png",dpi = 500)

fig, ax = plt.subplots()
ax.hist(HIST_DATA, bins = 200, color=colors2[:len(HIST_DATA)], label=CHI_PS)
ax.set_xlabel(r"$\frac{\rho}{\rho_{max}}$")
ax.set_ylabel("Count")
ax.set_yscale("log")
plt.suptitle("Polymer density distribution")
plt.tight_layout()
plt.legend()
plt.savefig(f"{rootdir}/histogram.png", dpi = 500)

h_file = open(f"{rootdir}/hist.pkl", "wb")
pickle.dump(HIST_DATA,h_file)
rho_file.close()


# fig, ax = plt.subplots()

# plt.suptitle("Cluster size distribution")

# ax[0].set_title("Cluster sizes")
# ax[0].plot(CHI_PS, polymer_rich, 'o-', label = "polymer-rich phase")
# ax[0].plot(CHI_PS, polymer_poor, 'o-', label = "polymer-poor phase")
# ax[0].xlabel(r"$\chi_{PS}$")
# ax[0].ylabel(r"$\frac{\rho_{pol} - <\rho_{pol}>}{\sigma_{pol}}$")

# ax[1].set_title("Average cluster size")
# ax[1].plot(CHI_PS, salt_rich, 'o-', label = "Salt in polymer-rich phase")
# ax[1].plot(CHI_PS, salt_rich, 'o-', label = "Salt in polymer-rich phase")
# ax[1].xlabel(r"$\chi_{PS}$")
# ax[1].ylabel(r"$\frac{\rho_{salt} - <\rho_{salt}>}{\sigma_{salt}}$")

# plt.tight_layout()
# plt.legend()
# plt.savefig(fname = f"{dir_name}/density.png",dpi = 500)


# data analysis - see if the ions partition

# 1. How to detect polymer-rich phase
# 2. Characterize the density
# 3. Data correlation 0.05 0.05 0.20

# Plot points in 3D C = rho


# polymer rich phase - x - slice
#
# For each slice get average polymer density


    # A = [1,+1.0, 1] #charged +
    # B = [2, 0, 0] #neutral
    # C = [3,-1.0, 1] #charged -
    # W = [4, 0, 1] #non-polar
    # CAT = [5,+1,0]
    # ANI = [6,-1,0]
    # CI = [7,1,0]
    # D = [8, 1, 0] #drude oscillator


# analyze energies, etc - pull the data file
# plot
# do some data analysis on that --

# spatial correlations, deviation from the mean - pointwise - how - 


# data_file = np.loadtxt('/home/jello/results/AB-block-21-Jun-2022-old/non-polar/np1.00/data.dat',skiprows = 2)
# time_step = data_file[:,0]
# UPe = data_file[:,1]
# UBond = data_file[:,2]

# Pressure = []
# for i in range(3,9):
#     Pressure.append(data_file[:,i])

# UGauss = []
# for i in range(9, 9+28):
#     UGauss.append(data_file[:,i])

# # plt.plot(time_step, UPe, label = 'UPe')
# # plt.legend()
# # #plt.show()
# # plt.plot(time_step, UBond, label = 'Ubond')
# # plt.legend()
# # #plt.show()

# types = ["A","B","C","W","CAT","ANI","CI","D"]

# count = 0
# for i in range(7):
#     for j in range(i,7):
#         print(f"{count}: {types[i]} {types[j]}")
#         count += 1

# for i,ugass in enumerate(UGauss):
#     if ugass[0] != 0 and i not in (18,21):
#         plt.plot(time_step,ugass, label = f"{i}")
# plt.legend()
# #plt.show()

# for i,p in enumerate(Pressure[:3]):
#     plt.plot(time_step,p, label = f"{i}")
# plt.legend()
# #plt.show()

# 18 is water-water -- solvent!

# they seem to condense because it minimizes the energy of 

# detect clusters
# measure cluser size



# to do -- cluster size detection


# cluster size vs cluser shape
# electrostatic energy ---





# Analysis to do:

# make a dedicated folder
# make an array of data
# chi p_rich p_poor s_rich s_poor
# plot density fileds per slice and the - go forward and backwards
# plot polymer concentrations and salt concentration as we vary chi - save the plots

# analyze pressure and energy over the course of simulation - extract the electrostatic energy
# cluster size distribution histogram
# density distribution histogram



