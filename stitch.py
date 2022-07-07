#!/usr/bin/env python3

import sys
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--head', default = "head.data")
parser.add_argument('--atoms', default = "atoms.data")
parser.add_argument('--file', default = "input.data")

args = parser.parse_args()

header_file = args.head
atom_file = args.atoms
file_name = args.file

with open(file=header_file,mode='r') as f:
    head = f.readlines()

with open(file=atom_file,mode='r') as f:
    atom_lines = f.readlines()

n_atoms = atom_lines[3]    

with open(file="tail.data",mode='r') as f:
    tail = f.readlines()

with open(file=file_name,mode='w') as fout:
    
    for line in head:
        fout.writelines(line)

    fout.writelines('\n')
    fout.writelines('Atoms\n')
    fout.writelines('\n')

    for atom in atom_lines[-int(n_atoms):]:
        line = atom.split()
        if len(line) == 7:
            fout.writelines(f"{line[0]} {line[2]} {line[1]}  {line[6]}  {line[3]}  {line[4]}  {line[5]}\n")

    for line in tail:
        fout.writelines(line)
