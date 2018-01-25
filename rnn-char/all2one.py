import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, help="Root directory of codes")
parser.add_argument("-o", "--outfile", type=str, help="Output file name")
args = parser.parse_args()

file = open(args.outfile, 'w')
for root, dirs, files in os.walk(args.dir):
    for filename in files:
        if filename.endswith(".formatted"):
            path = os.path.join(root, filename)
            data = open(path, 'r').read()
            file.write(data)

file.close()
