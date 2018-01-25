import numpy

class hackthon_data():
    def loaddata(self, filepath):
        file = open(filepath, 'r')
        self._alllines = file.readlines()
        file.close()
        self._index = 0
        self.totallen = len(self._alllines)
    
    def reset(self):
        self._index = 0
    
    # return a lable list and a data list
    # one data is also a list
    # example:
    # [0, 0, 1, 2] [[-58, -22, -1, -58, 51, -20, 36, 18, 13, 36, -12, 22, -7, 5], [-40, -62, 13, -35, 54, -3, 33, 44, -10, 29, -19, 16, -25, 11], [-25, -18, -33, -26, -34, -31, 0, -16, 0, -6, -8, -2, 0,
    # -1], [32, 1, 4, -8, -33, -63, -40, -25, 51, 0, 44, -3, 5, -1]]
    def getdata(self, batch_size):
        if self._alllines is None:
            print("you must open a data file first")
            return None

        if self._index + batch_size > self.totallen:
            # print("getdata meets end, no data returns")
            self._index = 0     # from begining again
        
        strlines = self._alllines[self._index:self._index + batch_size]
        self._index = self._index + batch_size

        ret_labels = []
        ret_vectors = []
        ret_points = []
        for strline in strlines:
            pointline = []
            numline = self.convert_one_line(strline)
            numline['points'] = sorted(numline['points'], key=lambda k: [k[1], k[0]])
            for point in numline['points']:
              pointline.append(point[0])
              pointline.append(point[1])
            if numline['value'] != None:
              ret_labels.append(numline['value'])
            # print("original data", numline)
            # self.normalize_points(numline['points'])
            # print("normalized data", numline)
            #ret_vectors.append(self.convert_points_to_vectors(numline['points']))
            ret_points.append(pointline)

        #return numpy.array(ret_labels, dtype='uint8'), numpy.array(ret_vectors, dtype='float32')
        return numpy.array(ret_labels, dtype='uint8'), numpy.array(ret_points, dtype='float32')

    # covert one line from string to dictionary like this:
    # {'value': 1, 'points': [[56, 100], [53, 82], [49, 56], [45, 25], [45, 9], [45, 3], [44, 1], [44, 0]]}
    def convert_one_line(self, line):
        # line = line[:-1]  # remove '\n' at end
        line.replace('\n', '')
        strlst = line.split(",")
        numlst = []
        for ss in strlst:
            numlst.append(int(ss))
        retvalue = {'points':[], 'value':None}
        if len(numlst) >=17:
            retvalue['value'] = numlst[16]
        for i in range(8):
            point = []
            point.append(numlst[i*2])
            point.append(numlst[i*2+1])
            retvalue['points'].append(point)
        # print(retvalue)
        return retvalue
    
    # convert a list of point to a list of vectors
    def convert_points_to_vectors(self, points):
        retvecs = []
        lastpoint = None
        for point in points:
            if lastpoint is not None:
                # vec = [point[0] - lastpoint[0], point[1] - lastpoint[1]]
                retvecs.append(point[0] - lastpoint[0])
                retvecs.append(point[1] - lastpoint[1])
                pass
            lastpoint = point
        # print(retvecs)
        return retvecs

    # normalize points, make sure it is in the center
    def normalize_points(self, points):
        max_x = 0
        min_x = 100
        max_y = 0
        min_y = 100
        for point in points:
            if point[0] > max_x:
                max_x = point[0]
            if point[0] < min_x:
                min_x = point[0]
            if point[1] > max_y:
                max_y = point[1]
            if point[1] < min_y:
                min_y = point[1]
        # print(max_x, min_x, max_y, min_y)
        # print(max_x, min_x)
        for point in points:
            if max_x == min_x:
                point[0] = 50
            else:
                point[0] = round((point[0] - min_x) / (max_x - min_x) * 100)
            if max_y == min_y:
                point[1] = 50
            else:
                point[1] = round((point[1] - min_y) / (max_y - min_y) * 100)
        # print(points)



if __name__ == '__main__':
    filepath = "Train.dat"
    data = hackthon_data()
    data.loaddata(filepath)
    labels, vectors = data.getdata(10)
    print(labels, vectors)
    print(type(vectors), labels.dtype, vectors.dtype)
    print(vectors.shape)
    # labels, vectors = data.getdata(2)
    # print(labels, vectors)
    # labels, vectors = data.getdata(2)
    # print(labels, vectors)
    # line =  ret[2]
    # numline = data.convert_one_line(line)
    # print(numline)
    # data.normalize_points(numline['points'])
    # print(numline)
    # vecline = data.convert_points_to_vectors(numline['points'])
    # print(vecline)
