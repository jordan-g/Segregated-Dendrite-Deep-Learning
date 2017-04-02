# encoding=utf8
'''
Code for loading MNIST data set files from binary files.
Adapted from: https://github.com/johnmyleswhite/MNIST.jl

Copyright (C) 2017 Jordan Guerguiev

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import numpy as np
import struct
import sys

if sys.version_info >= (3,):
    xrange = range

# Constants
IMAGEOFFSET = 16
LABELOFFSET = 8
NROWS       = 28
NCOLS       = 28
TRAINIMAGES = "train-images.idx3-ubyte"
TRAINLABELS = "train-labels.idx1-ubyte"
TESTIMAGES  = "t10k-images.idx3-ubyte"
TESTLABELS  = "t10k-labels.idx1-ubyte"

def imageheader(filename):
    with open(filename, 'rb') as io:
        magic_number = struct.unpack('>i', io.read(4))[0]
        total_items  = struct.unpack('>i', io.read(4))[0]
        nrows = struct.unpack('>i', io.read(4))[0]
        ncols = struct.unpack('>i', io.read(4))[0]
    return magic_number, int(total_items), int(nrows), int(ncols)

def labelheader(filename):
    with open(filename, 'rb') as io:
        magic_number = struct.unpack('>i', io.read(4))[0]
        total_items  = struct.unpack('>i', io.read(4))[0]
    return magic_number, int(total_items)

def getimage(filename, index):
    with open(filename, 'rb') as io:
        io.seek(IMAGEOFFSET + NROWS * NCOLS * index)
        image = np.empty((NROWS, NCOLS))
        for i in np.nditer(image, op_flags=['readwrite']):
            i[...] = struct.unpack('B', io.read(1))[0]
    return image

def getlabel(filename, index):
    with open(filename, 'rb') as io:
        io.seek(LABELOFFSET + index)
        label = struct.unpack('B', io.read(1))[0]
    return label

def trainimage(index): return getimage(TRAINIMAGES, index)
def trainlabel(index): return getlabel(TRAINLABELS, index)
def testimage(index): return getimage(TESTIMAGES, index)
def testlabel(index): return getlabel(TESTLABELS, index)
def trainfeatures(index): return trainimage(index).flatten(order='F')
def testfeatures(index): return testimage(index).flatten(order='F')

def traindata():
    _, nimages, nrows, ncols = imageheader(TRAINIMAGES)
    features = np.empty((nrows * ncols, nimages))
    labels   = np.empty(nimages)
    for index in xrange(nimages):
        features[:, index] = trainfeatures(index)
        labels[index] = trainlabel(index)
    return features, labels

def testdata():
    _, nimages, nrows, ncols = imageheader(TESTIMAGES)
    features = np.empty((nrows * ncols, nimages))
    labels   = np.empty(nimages)
    for index in xrange(nimages):
        features[:, index] = testfeatures(index)
        labels[index] = testlabel(index)
    return features, labels