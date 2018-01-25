#!/usr/bin/python
# -*- coding: utf-8 -*- 

import sys
import re
import io

def pre_process(input_file, output_file):
    multi_version = re.compile('-\{|zh-hans|zh-hant|zh-cn|\}-')
    punctuation = re.compile("[-~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】、；‘：“”，。、《》？「『」』]")
    with io.open(output_file, mode = 'w', encoding = 'utf-8') as outfile:
        with io.open(input_file, mode = 'r', encoding ='utf-8') as infile:
            for line in infile:
                if line.startswith('<doc') or line.startswith('</doc'):
                    #print(line)
                    continue
                line = multi_version.sub('', line)
                line = punctuation.sub(' ', line)
                outfile.write(line)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file output_file")
        sys.exit()
    input_file, output_file = sys.argv[1], sys.argv[2]
    pre_process(input_file, output_file)
