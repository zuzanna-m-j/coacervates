#!/usr/bin/env python3

"""
File: gen-input.py
Author: Zuzanna M. J.
Description:
***
Code generates initial configuration to be run using gpu-tild code.
Command-line arhuments can be used to modify the input. Default values are provided.
Output can be analyzed using analyze-data.py provided in the package.

Command-line arguments:

 - 
 -
 -
 -


"""
from ast import arg
import random
import numpy as np
import copy
import os
import sys
from numpy import pi, sqrt
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--wlc', default = False, type = bool)
parser.add_argument('--flat', default = False, type = bool)
parser.add_argument('--water',default = True, type = bool)
parser.add_argument('--polar',default = False, type = bool)
parser.add_argument('--midpush',default = False, type = bool)

parser.add_argument('--salt', default = 0.0, type = float)
parser.add_argument('--chips', default = 1.0, type = float)
parser.add_argument('--chipi', default = 0.0, type = float)
parser.add_argument('--chibs', default = 0.0, type = float)

parser.add_argument('--lx', default = 20.0, type = float)
parser.add_argument('--ly', default = 20.0, type = float)
parser.add_argument('--lz', default = 100.0, type = float)

parser.add_argument('--zmin', default = 50, type = float)
parser.add_argument('--zmax', default = 75, type = float)

parser.add_argument('--Nx', default = 45, type = int)
parser.add_argument('--Ny', default = 45, type = int)
parser.add_argument('--Nz', default = 215, type = int)

parser.add_argument('--scale', default = 1.0, type = float)
parser.add_argument('--dim', default = 3, type = int)


args = parser.parse_args()

POLAR = args.polar
FLAT = args.flat
WATER = args.water
WLC = args.wlc
DIM = args.dim
MIDPUSH = args.midpush

SALT = args.salt
chi_ps = args.chips
chi_pi = args.chipi
chi_bs = args.chibs

lx = args.lx * args.scale
ly = args.ly * args.scale
lz = args.lz * args.scale

zmin = args.zmin
zmax = args.zmax

Nx = int(args.Nx * args.scale)
Ny = int(args.Ny * args.scale)
Nz = int(args.Nz * args.scale)

N_a = 25
N_b = 0
N_c = 25
N = N_a + N_b + N_c
seq = N_a * "A" + N_b * "B" + N_c * "C"

box_dim = [lx, ly, lz]

if FLAT == True:
    box_vol = lx * ly
else:
    box_vol = lx * ly * lz


rho0 = 3.0
phi = 0.045
kappaN = 5 * 50
kappa = kappaN/N

n_pol = int(phi * box_vol * rho0/N)
n_ci =  int(phi * box_vol * rho0 * (N_a + N_b)/N)
n_sol = int(rho0 * box_vol - N * n_pol - n_ci)
CHI = [
    [0,0,0,chi_ps,chi_pi,chi_pi,chi_pi],
    [0,0,0,chi_bs,chi_pi,chi_pi,chi_pi],
    [0,0,0,chi_ps,chi_pi,chi_pi,chi_pi],
    [chi_ps, chi_bs, chi_ps,0,0,0,0],
    [chi_pi, chi_pi, chi_pi,0,0,0,0],
    [chi_pi, chi_pi, chi_pi,0,0,0,0],
    [chi_pi, chi_pi, chi_pi,0,0,0,0]]

molecule_types = 1
atom_count = 1
mol_count = 1
bond_count = 1
angle_count = 0

if POLAR == True:

    A =   [1,+1, 1]
    B =   [2, 0, 0]
    C =   [3,-1, 1]
    W =   [4, 0, 0] 
    CAT = [5,+1, 0]
    ANI = [6,-1, 0]
    CI =  [7, 1, 0]
    D =   [8, 1, 0]

else:
    A =   [1,+1, 0]
    B =   [2, 0, 0]
    C =   [3,-1, 0]
    W =   [4, 0, 0] 
    CAT = [5,+1, 0]
    ANI = [6,-1, 0]
    CI =  [7, 1, 0]
    D =   [8, 1, 0]

types = [A,B,C,W,CAT,ANI,CI,D]
particle_types = len(types)

aii = kappa/(2 * rho0)
Aij = np.zeros((particle_types-1,particle_types-1))
g_count = 0
for i in range(particle_types-1):
    for j in range(i,particle_types-1):
        if i == j:
            Aij[i][j] = aii
            g_count += 1
        elif  i != j:
            Aij[i][j] = CHI[i][j]/rho0 + 2.0 * aii
            g_count += 1

if WLC == True:
    angle_types = 1
else:
    angle_types = 0

bond_types = 3

properties = []
bonds = []
mol_angles = []
angles = []

q_plus = 0
q_minus = 0

for m_num in range(n_pol):
    mol_ang = []
    for chain_pos in range(N):
        m = seq[chain_pos]

        if m == 'A':
            m = A[0] - 1
        elif m == 'B':
            m = B[0] - 1
        elif m == 'C':
            m = C[0] - 1

        props = [atom_count,mol_count,types[m][0]]
        qm = types[m][1]
        dm = types[m][2]

        if dm == 1:
            if qm == 1.0:
                atom_charge =  qm
                q_minus += 1
            elif qm == -1.0:
                atom_charge = qm
                q_plus += 1

            if qm ==0:
                atom_charge = 0

            if atom_charge > 0:
                drude_charge = -1/2
                atom_charge +=  1/2

            elif atom_charge < 0:
                drude_charge = atom_charge
                drude_charge -= 1/2
                atom_charge =  1/2

            elif atom_charge == 0:
                atom_charge = 1/2
                drude_charge = -1/2
            
            props.append(atom_charge)


            if chain_pos == 0:
                for xyz in range(3):
                    coord = np.random.uniform(0,box_dim[xyz])
                    props.append(coord)
                if FLAT == True:
                    props[-1] = 0.0
            else:
                theta = random.uniform(-np.pi, np.pi)
                phi = random.uniform(- 2 * np.pi, 2 * np.pi)
                x = 1.0 * np.cos(phi)*np.sin(theta) + properties[-1][4]
                y = 1.0 * np.sin(phi)*np.sin(theta) + properties[-1][5]
                z = 1.0 * np.cos(theta) + properties[-1][6]
                if FLAT == True:
                    z = 0.0
                props.append(x)
                props.append(y)
                props.append(z)
            
            # add atom properties to the list
            properties.append(copy.deepcopy(props))
            mol_ang.append(atom_count)

            # drude bond - type 2
            bonds.append([bond_count,2,atom_count,atom_count+1])
            bond_count += 1

            # regular bond - 1
            if chain_pos != (N-1):
                bonds.append([bond_count,1,atom_count,atom_count+2])
                bond_count += 1

            # advance the atom count
            atom_count += 1

            # add the drude oscilator

            dx = properties[-1][4]
            dy = properties[-1][5]
            dz = properties[-1][6]

            props = [atom_count,mol_count,D[0],drude_charge,dx,dy,dz]
            properties.append(copy.deepcopy(props))
            atom_count += 1

        else:
            atom_charge = qm
            props.append(atom_charge)

            if chain_pos == 0:
                for xyz in range(3):
                    coord = np.random.uniform(0,box_dim[xyz])
                    props.append(coord)
                if FLAT == True:
                    props[-1] == 0.0
            else:
                theta = random.uniform(-np.pi, np.pi)
                phi = random.uniform(- 2 * np.pi, 2 * np.pi)
                x = 1.0 * np.cos(phi)*np.sin(theta) + properties[-1][4]
                y = 1.0 * np.sin(phi)*np.sin(theta) + properties[-1][5]
                z = 1.0 * np.cos(theta) + properties[-1][6]
                if FLAT == True:
                    z = 0.0
                props.append(x)
                props.append(y)
                props.append(z)
            
            properties.append(copy.deepcopy(props))
            mol_ang.append(atom_count)

            if chain_pos != (N-1):
                bonds.append([bond_count,1,atom_count,atom_count+1])
                bond_count += 1
            atom_count += 1
            
    mol_angles.append(copy.deepcopy(mol_ang))
    mol_count += 1

for i in range(n_ci//2):
    props = [atom_count,mol_count,CI[0], 1]
    for xyz in range(3):
        coord = np.random.uniform(0,1) * box_dim[xyz]
        props.append(coord)   
    properties.append(copy.deepcopy(props))
    atom_count += 1
    mol_count += 1

for i in range(n_ci//2):
    props = [atom_count,mol_count,CI[0], -1]
    for xyz in range(3):
        coord = np.random.uniform(0,1) * box_dim[xyz]
        props.append(coord)     
    properties.append(copy.deepcopy(props))
    atom_count += 1
    mol_count += 1

# if SALT == True:
#     salt_charge = 0
#     for _ in range(N_cat):
#         props = [atom_count,mol_count,CAT[0], CAT[1]]
#         for xyz in range(DIM):
#             coord = np.random.uniform(0,1) * box_dim[xyz]
#             props.append(coord)     
#         properties.append(copy.deepcopy(props))
#         atom_count += 1
#         mol_count += 1
#         salt_charge += CAT[1]
#     for _ in range(N_anion):
#         props = [atom_count,mol_count,ANI[0], ANI[1]]
#         for xyz in range(DIM):
#             coord = np.random.uniform(0,1) * box_dim[xyz]
#             props.append(coord)     
#         properties.append(copy.deepcopy(props))
#         atom_count += 1
#         mol_count += 1
#         salt_charge +=  ANI[1]
#     print(f"Total salt charge: {salt_charge}")

if WATER == True:
    for _ in range(n_sol):
        props = [atom_count,mol_count,W[0], 1/2]
        for xyz in range(3):
            coord = np.random.uniform(0,1) * box_dim[xyz]
            props.append(coord)     
        properties.append(copy.deepcopy(props))

        bonds.append([bond_count,3,atom_count,atom_count+1])
        bond_count += 1
        atom_count += 1
        dx = properties[-1][4]
        props = [atom_count,mol_count,D[0],-1/2,dx,dx,dx]
        properties.append(copy.deepcopy(props))
        atom_count += 1
        mol_count += 1

if WLC == True:
    #process angles
    for mol in mol_angles:
        for i in range(len(mol)-2):
            angles.append([angle_count,1, mol[i], mol[i+1], mol[i+2]])
            angle_count += 1

with open("head.data", 'w') as fout:
    fout.writelines("Madatory string --> First rule of programing: if it works then don't touch it!\n\n")
    fout.writelines(f'{atom_count - 1} atoms\n')
    fout.writelines(f'{bond_count - 1} bonds\n')
    if WLC == True:
        fout.writelines(f'{angle_count - 1} angles\n')
    else:
        fout.writelines(f'{0} angles\n')
    fout.writelines('\n')
    fout.writelines(f'{particle_types} atom types\n')
    fout.writelines(f'{bond_types} bond types\n')
    fout.writelines(f'{angle_types} angle types\n')
    fout.writelines('\n')
    fout.writelines(f'0.000 {box_dim[1]} xlo xhi\n')
    fout.writelines(f'0.000 {box_dim[1]} ylo yhi\n')
    fout.writelines(f'0.000 {box_dim[2]} zlo zhi\n')
    fout.writelines('\n')
    fout.writelines('Masses\n')
    fout.writelines('\n')
    for i in range(len(types)):
        fout.writelines(f'{i + 1} {1.000} \n')

with open('atoms.data','w') as fout:
    for i in range(9):
        fout.writelines(f"{atom_count - 1}\n")
    for line in properties:
        fout.writelines(f"{line[0]} {line[2]} {line[1]} {line[4]} {line[5]} {line[6]} {line[3]}\n")

with open('tail.data', 'w') as fout:       
    fout.writelines('\n')
    fout.writelines('Bonds\n')
    fout.writelines('\n')
    for line in bonds:
        fout.writelines(f"{line[0]} {line[1]}  {line[2]} {line[3]}\n")
    if WLC == True:
        fout.writelines('\n')
        fout.writelines('Angles\n')
        fout.writelines('\n')
        for line in angles:
            fout.writelines(f"{line[0]} {line[1]}  {line[2]} {line[3]} {line[4]}\n")


with open("head2.data", 'w') as fout:
    fout.writelines("Madatory string --> First rule of programing: if it works then don't touch it!\n\n")
    fout.writelines(f'{atom_count - 1} atoms\n')
    fout.writelines(f'{bond_count - 1} bonds\n')
    if WLC == True:
        fout.writelines(f'{angle_count - 1} angles\n')
    else:
        fout.writelines(f'{0} angles\n')
    fout.writelines('\n')
    fout.writelines(f'{particle_types} atom types\n')
    fout.writelines(f'{bond_types} bond types\n')
    fout.writelines(f'{angle_types} angle types\n')
    fout.writelines('\n')
    fout.writelines(f'0.000 {box_dim[0]} xlo xhi\n')
    fout.writelines(f'0.000 {box_dim[1]} ylo yhi\n')
    fout.writelines(f'0.000 {box_dim[2]} zlo zhi\n')
    fout.writelines('\n')
    fout.writelines('Masses\n')
    fout.writelines('\n')
    for i in range(len(types)):
        fout.writelines(f'{i + 1} {1.000} \n')

with open('atoms.data','w') as fout:
    for i in range(9):
        fout.writelines(f"{atom_count - 1}\n")
    for line in properties:
        fout.writelines(f"{line[0]} {line[2]} {line[1]} {line[4]} {line[5]} {line[6]} {line[3]}\n")

with open('tail.data', 'w') as fout:       
    fout.writelines('\n')
    fout.writelines('Bonds\n')
    fout.writelines('\n')
    for line in bonds:
        fout.writelines(f"{line[0]} {line[1]}  {line[2]} {line[3]}\n")
    if WLC == True:
        fout.writelines('\n')
        fout.writelines('Angles\n')
        fout.writelines('\n')
        for line in angles:
            fout.writelines(f"{line[0]} {line[1]}  {line[2]} {line[3]} {line[4]}\n")

input_file = f"""Dim {DIM}

max_steps 1000001
log_freq 1000
binary_freq 10000
traj_freq 500000
pmeorder 1

charges {55:7f} {0.5}

delt {0.005:7f}

read_data input.data
integrator all GJF

Nx {Nx}
Ny {Ny}
Nz {Nz}

bond 1 harmonic {1.0:7f} {0.0:7f}
bond 2 harmonic {2.5:7f} {0.0:7f}
bond 3 harmonic {2.5:7f} {0.0:7f}

"""

if MIDPUSH == True:
    input_file += f"""group polya type 1
group polyc type 3
extraforce polya midpush 0.05
extraforce polyc midpush 0.05

"""

if WLC == True:
    input_file += f"""angle 1 wlc {1.0:7f}
    """

input_file += f"\nn_gaussians {g_count}\n"
for i in range(particle_types-1):
    for j in range(i,particle_types-1):
        input_file += f"gaussian {i+1} {j+1} {Aij[i][j]}  {1.000}\n"

with open('input', 'w') as fout:       
    fout.writelines(input_file)



input_file = f"""Dim {DIM}

max_steps 1500001
log_freq 1000
binary_freq 10000
traj_freq 500000
pmeorder 1

charges {55:7f} {0.5}

delt {0.005:7f}

read_data input.data
integrator all GJF

Nx {Nx}
Ny {Ny}
Nz {Nz}

bond 1 harmonic {1.0:7f} {0.0:7f}
bond 2 harmonic {2.5:7f} {0.0:7f}
bond 3 harmonic {2.5:7f} {0.0:7f}

"""

if WLC == True:
    input_file += f"""angle 1 wlc {1.0:7f}
    """

input_file += f"\nn_gaussians {g_count}\n"
for i in range(particle_types-1):
    for j in range(i,particle_types-1):
        input_file += f"gaussian {i+1} {j+1} {Aij[i][j]}  {1.000}\n"

with open('input2', 'w') as fout:       
    fout.writelines(input_file)



file_name = os.getcwd()

descr = f"""File: {file_name}
Polymer density: {N * n_pol/box_vol}
Midpush = {MIDPUSH}
Polymer number {n_pol}
Solvent: {n_sol}
Counter ions: {n_ci}
Number density: {(n_pol * N + n_sol + n_ci)/box_vol}
Polymer volume fraction: {N * n_pol/(n_pol * N + n_sol + n_ci)}
Polarity: {POLAR}
Salt: {SALT}
Small: {FLAT}
Chi_PS: {chi_ps}
Chi_BS: {chi_bs}
Chi_PI: {chi_pi}
BOX: {lx} {ly} {lz}
N_GRID {Nx} {Ny} {Nz}
DIM: {DIM}

"""
with open("../../info.txt", 'a+') as f:
    f.writelines(descr)
print(descr)

# group polya type 1
# group polyb type 2
# group polyc type 3
# extraforce polya midpush 0.05
# extraforce polyb midpush 0.05
# extraforce polyc midpush 0.05
