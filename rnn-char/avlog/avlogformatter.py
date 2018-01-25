#!/usr/bin/python
# -*- coding: utf-8 -*- 

import sys
import re
import io

def pre_process(input_file, output_file):
    regex1 = re.compile('^\d.*\d\s(?=a|\[)')
    regex2 = re.compile('^\[.*\]\s{2}')
    with io.open(output_file, mode = 'w', encoding = 'utf-8') as outfile:
        with io.open(input_file, mode = 'r', encoding = 'utf-8') as infile:
            for line in infile:
                line = regex1.sub('', line)
                line = regex2.sub('', line)
                outfile.write(line)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python avlogformatter.py input_file")
        sys.exit()
    input_file = sys.argv[1]
    pre_process(input_file, input_file+'.formatted')
