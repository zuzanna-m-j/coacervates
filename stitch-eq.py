#!/usr/bin/env python3

import sys
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--head', default = "head.data")
parser.add_argument('--atoms', default = "atoms.data")
parser.add_argument('--sol', default = "sol.data")
parser.add_argument('--info', default = "info.data")
parser.add_argument('--file', default = "input.data")

args = parser.parse_args()

header_file = args.head
atom_file = args.atoms
sol_file = args.sol
info_file = args.info
file_name = args.file


with open(file=header_file,mode='r') as f:
    head = f.readlines()

with open(file=atom_file,mode='r') as f:
    atom_lines = f.readlines()

with open(file=sol_file,mode='r') as f:
    sol_lines = f.readlines()

with open(file=info_file,mode = 'r') as f:
    info_lines = f.readlines()

n_atoms = info_lines[0].strip().split()
n_atoms = int(n_atoms[1])

n_pol = info_lines[1].strip().split()
n_pol = int(n_pol[1])


with open(file="tail.data",mode='r') as f:
    tail = f.readlines()

with open(file=file_name,mode='w') as fout:
    
    for line in head:
        fout.writelines(line)

    fout.writelines('\n')
    fout.writelines('Atoms\n')
    fout.writelines('\n')

    for atom in atom_lines[-n_atoms:-n_atoms + n_pol]:
        line = atom.split()
        if len(line) == 7:
            fout.writelines(f"{line[0]} {line[2]} {line[1]}  {line[6]}  {line[3]}  {line[4]}  {line[5]}\n")

    for atom in sol_lines:
        line = atom.split()
        if len(line) == 7:
            fout.writelines(f"{line[0]} {line[2]} {line[1]}  {line[6]}  {line[3]}  {line[4]}  {line[5]}\n")

    for line in tail:
        fout.writelines(line)
