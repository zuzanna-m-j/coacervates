#!/usr/bin/env python3

from ast import arg
import random
import numpy as np
import copy
import os
import sys
from numpy import pi, sqrt
import argparse

EXPLICIT_SOLVENT = False
SALT = False
WLC = False
INTERACT = False

if "-wlc" in sys.argv:
    WLC = True
if "-salt" in sys.argv:
    SALT = True
if "-water" in sys.argv:
    EXPLICIT_SOLVENT = True
if "-interact" in sys.argv:
    INTERACT = True

parser = argparse.ArgumentParser()

parser.add_argument('--wlc', default = False, type = bool)
parser.add_argument('--small', default = False, type = bool)
parser.add_argument('--salt', default = 0.0, type = float)
parser.add_argument('--water',default = True, type = bool)
parser.add_argument('--polar',default = True, type = bool)
parser.add_argument('--chips', default = 0.0, type = float)
parser.add_argument('--chipi', default = 0.0, type = float)
parser.add_argument('--chibs', default = 0.0, type = float)

args = parser.parse_args()

POLAR = arg.polar
chi_ps = args.chips
chi_pi = args.chipi
chi_bs = args.chibs

N_a = 25
N_b = 0
N_c = 25
N = N_a + N_b + N_c
seq = N_a * "A" + N_b * "B" + N_c * "C"

dim = 3
lx = 20
ly = 20
lz = 100

Nx = 47
Ny = 47
Nz = 215

box_dim = [lx, ly, lz]
box_vol = lx * ly * lz

rho0 = 3.0
phi = 0.045
kappaN = 5 * 50
kappa = kappaN/N

n_pol = int(0.045 * box_vol * rho0/N)
n_ci =  int(0.045 * box_vol * rho0 * (N_a + N_b)/N)
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

if WLC == True:
    angle_types = 1
else:
    angle_types = 0

properties = []
bonds = []
mol_angles = []
angles = []

N_sol = int(((1.0 - np.sum(phi)) * box_vol))
if EXPLICIT_SOLVENT == True:
    bond_types = 3

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
                for xyz in range(dim-1):
                    coord = np.random.uniform(0,box_dim[xyz])
                    props.append(coord)
                if SMALL == True:
                    z = np.random.uniform(2/5 * box_dim[2], 3/5 *box_dim[2])
                else:
                    z = np.random.uniform(0,box_dim[2])
                props.append(z)
            else:
                if properties[-1][2] != D[0]:
                    theta = random.uniform(-np.pi, np.pi)
                    phi = random.uniform(- 2 * np.pi, 2 * np.pi)
                    x = 1.0 * np.cos(phi)*np.sin(theta) + properties[-1][4]
                    y = 1.0 * np.sin(phi)*np.sin(theta) + properties[-1][5]
                    z = 1.0 * np.cos(theta) + properties[-1][6]
                    if SMALL == True:
                        while 2/5 * box_dim[2] > z > 3/5 *box_dim[2]:
                            z = 1.0 * np.cos(theta) + properties[-1][6]
                    props.append(x)
                    props.append(y)
                    props.append(z)

                else:
                    theta = random.uniform(-np.pi, np.pi)
                    phi = random.uniform(- 2 * np.pi, 2 * np.pi)
                    x = 1.0 * np.cos(phi)*np.sin(theta) + properties[-2][4]
                    y = 1.0 * np.sin(phi)*np.sin(theta) + properties[-2][5]
                    z = 1.0 * np.cos(theta) + properties[-2][6]
                    if SMALL == True:
                        while 2/5 * box_dim[2] > z > 3/5 *box_dim[2]:
                            z = 1.0 * np.cos(theta) + properties[-1][6]                    
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

            theta = random.uniform(-np.pi, np.pi)
            phi = random.uniform(- 2 * np.pi, 2 * np.pi)
            dx = 1.0/2 * np.cos(phi)*np.sin(theta) + properties[-1][4]
            dy = 1.0/2 * np.sin(phi)*np.sin(theta) + properties[-1][5]
            dz = 1.0/2 * np.cos(theta) + properties[-1][6]

            props = [atom_count,mol_count,D[0],drude_charge,dx,dy,dz]
            properties.append(copy.deepcopy(props))
            atom_count += 1

        

        else:
            atom_charge = qm
            props.append(atom_charge)

            if chain_pos == 0:
                for xyz in range(dim-2):
                    coord = np.random.uniform(0,box_dim[xyz])
                    props.append(coord)
                if SMALL == True:
                    z = np.random.uniform(2/5 * box_dim[2], 3/5 *box_dim[2])
                else:
                    z = np.random.uniform(0,box_dim[2])
                props.append(z)
            else:

                if properties[-1][2] != D[0]:
                    theta = random.uniform(-np.pi, np.pi)
                    phi = random.uniform(- 2 * np.pi, 2 * np.pi)
                    x = 1.0 * np.cos(phi)*np.sin(theta) + properties[-1][4]
                    y = 1.0 * np.sin(phi)*np.sin(theta) + properties[-1][5]
                    z = 1.0 * np.cos(theta) + properties[-1][6]
                    if SMALL == True:
                        while 2/5 * box_dim[2] > z > 3/5 *box_dim[2]:
                            z = 1.0 * np.cos(theta) + properties[-1][6] 
                    props.append(x)
                    props.append(y)
                    props.append(z)

                else:
                    theta = random.uniform(-np.pi, np.pi)
                    phi = random.uniform(- 2 * np.pi, 2 * np.pi)
                    x = 1.0 * np.cos(phi)*np.sin(theta) + properties[-2][4]
                    y = 1.0 * np.sin(phi)*np.sin(theta) + properties[-2][5]
                    z = 1.0 * np.cos(theta) + properties[-2][6]
                    if SMALL == True:
                        while 2/5 * box_dim[2] > z > 3/5 *box_dim[2]:
                            z = 1.0 * np.cos(theta) + properties[-1][6]  
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
    for xyz in range(dim):
        coord = np.random.uniform(0,1) * box_dim[xyz]
        props.append(coord)     
    properties.append(copy.deepcopy(props))
    atom_count += 1
    mol_count += 1

for i in range(n_ci//2):
    props = [atom_count,mol_count,CI[0], -1]
    for xyz in range(dim):
        coord = np.random.uniform(0,1) * box_dim[xyz]
        props.append(coord)     
    properties.append(copy.deepcopy(props))
    atom_count += 1
    mol_count += 1

# if SALT == True:
#     salt_charge = 0
#     for _ in range(N_cat):
#         props = [atom_count,mol_count,CAT[0], CAT[1]]
#         for xyz in range(dim):
#             coord = np.random.uniform(0,1) * box_dim[xyz]
#             props.append(coord)     
#         properties.append(copy.deepcopy(props))
#         atom_count += 1
#         mol_count += 1
#         salt_charge += CAT[1]
#     for _ in range(N_anion):
#         props = [atom_count,mol_count,ANI[0], ANI[1]]
#         for xyz in range(dim):
#             coord = np.random.uniform(0,1) * box_dim[xyz]
#             props.append(coord)     
#         properties.append(copy.deepcopy(props))
#         atom_count += 1
#         mol_count += 1
#         salt_charge +=  ANI[1]
#     print(f"Total salt charge: {salt_charge}")

if EXPLICIT_SOLVENT == True:
    for _ in range(N_sol):
        props = [atom_count,mol_count,W[0], 1/2]
        for xyz in range(dim):
            coord = np.random.uniform(0,1) * box_dim[xyz]
            props.append(coord)     
        properties.append(copy.deepcopy(props))

        bonds.append([bond_count,3,atom_count,atom_count+1])
        bond_count += 1
        atom_count += 1
        dx = 1.0/2 + properties[-1][4]
        props = [atom_count,mol_count,D[0],-1/2,dx,dy,dz]
        properties.append(copy.deepcopy(props))
        atom_count += 1
        mol_count += 1

if WLC == True:
    #process angles
    for mol in mol_angles:
        for i in range(len(mol)-2):
            angles.append([angle_count,1, mol[i], mol[i+1], mol[i+2]])
            angle_count += 1

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

input_file = f"""Dim 3

max_steps 2000001
log_freq 5000
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
for i in range(particle_types):
    for j in range(i,particle_types):
        input_file += f"gaussian {i+1} {j+1} {Aij[i][j]}  {1.000}\n"

with open('input', 'w') as fout:       
    fout.writelines(input_file)
