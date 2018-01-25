import numpy as np
import fileinput

def gencosin(poslist):
    coslist = []
    n = (len(poslist) - 1) / 2
    for i in xrange(n-2):
        ii = i*2
        #print(poslist[ii:ii+2])
        #print(poslist[ii+2:ii+4])
        #print(poslist[ii+4:ii+6])

        vec1 = np.array(poslist[ii:ii+2]) - np.array(poslist[ii+2:ii+4])
        vec2 = np.array(poslist[ii+4:ii+6]) - np.array(poslist[ii+2:ii+4])
        #print("vec1 %s vec2 %s\n" %(vec1, vec2))

        cos = vec1.dot(vec2) / (np.sqrt(vec1.dot(vec1)) * np.sqrt(vec2.dot(vec2)))
        coslist.append(cos)
    
    label = int(poslist[-1])
    return coslist, label

def loaddata(filepath):
    cosarray = []
    labelarray = []
    for line in fileinput.input(filepath):
        #print(line)
        line = line.split(',')
        #print(line)
        line = map(float, line)
        #print("%s\n" % line)
        coslist, label = gencosin(line)
        cosarray.append(coslist)
        labelarray.append(label)
        #print("cos %s\n" % coslist)
    return np.array(cosarray), np.array(labelarray)
