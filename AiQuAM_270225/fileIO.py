#!/usr/bin/python3

# fileIO: utility functions for input and output of volume files.
#
# Copyright (C) 2015  J-P Suuronen
#

#import array as arr
import numpy as np

import os
#~ from debugUtils import keyboard


typeDict = {\
    np.dtype('bool'):'binary',\
    np.dtype('uint8'):'uint8',\
    np.dtype('uint16'):'uint16',\
    np.dtype('uint32'):'uint32',\
    np.dtype('int8'):'int8',\
    np.dtype('int16'):'int16',\
    np.dtype('int32'):'int32',\
    np.dtype('float32'):'single',\
    np.dtype('float64'):'double',\
    'boolean':np.dtype('bool'),\
    'binary':np.dtype('bool'),\
    'uint8':np.dtype('uint8'),\
    'uint16':np.dtype('uint16'),\
    'uint32':np.dtype('uint32'),\
    'int8':np.dtype('int8'),\
    'int16': np.dtype('int16'),\
    'int32': np.dtype('int32'),\
    'float32': np.dtype('float32'),\
    'float': np.dtype('float32'),\
    'single': np.dtype('float32'),\
    'float64': np.dtype('float64'),\
    'double': np.dtype('float64'),\
}



def parseDtype(dtype):
    
    if dtype=='char':
        dtype = 'c'
    elif dtype=='int8':
        dtype = 'b'
    elif dtype=='uint8':
        dtype = 'B'
    elif dtype=='int16':
        dtype = 'h'
    elif dtype=='uint16':
        dtype = 'H'
    elif dtype=='int32':
        dtype = 'l'
    elif dtype=='uint32':
        dtype = 'L'
    elif (dtype=='single' or dtype=='float32'):
        dtype = 'f'
    elif (dtype=='float' or dtype=='double' or dtype=='float64'):
        dtype = 'd'
    elif not((dtype[-1].lower() == 'c' or 
            dtype[-1].lower() == 'b' or
            dtype[-1].lower() == 'h' or 
            dtype[-1].lower() == 'i' or 
            dtype[-1].lower() == 'l' or 
            dtype[-1].lower() == 'f' or 
            dtype[-1].lower() == 'u' or 
            dtype[-1].lower() == 'd') and
            (len(dtype) == 1 or
            (len(dtype) == 2 and
            (dtype[0] == "<" or 
            dtype[0] == ">")))):
        raise Exception('Invalid datatype!, %s' % dtype)
    
    return dtype


def getVolumeParamsFromFilename(filename,bufferType=None):
    
    #Why is bufferType an argument??? I guess it's so that you can call this with a bufferType in case you  already know it will not find one, and still end up with something valid as output and not have to check again if it is None. Should the bufferType given as argument override the one in the file name?
    
    if not isinstance(filename, str):
        raise TypeError('File name must be a string')

    nameStart=filename.rsplit('.',1)[0].lower() #Split at the last period, make sure everything is in lowercase
                           
    tokens=nameStart.replace('x','_').split('_')[-4:]
    
    if len(tokens)!=4:
        raise ValueError("Filename {} cannot be used to deduce volume geometry".format(filename))
    
    try: 
        bufferType = typeDict[tokens[0]]
    except KeyError:
        try:
            bufferType = typeDict[tokens[-1]]
        except KeyError:
            pass
    
    try:
        dims=tuple([int(sr) for sr in tokens[:-1]])
    except ValueError:
        try:
            dims=tuple([int(sr) for sr in tokens[1:]])
        except ValueError:
            dims = None               
    
    return dims,bufferType


def guessDataType(fileName,dims=None):
    bufferType = None
    if dims is None:
        dims,bufferType = getVolumeParamsFromFilename(fileName)
    if bufferType is None:
        fileSize = os.path.getsize(fileName)
        numVoxels = np.prod(dims)
        if fileSize % numVoxels:
            raise ValueError("File size is not compatible with the given dimensions")
        else:
            bytesPerVoxel = fileSize//numVoxels
            if bytesPerVoxel==1:
                bufferType = typeDict['uint8']
            elif bytesPerVoxel==2:
                bufferType = typeDict['uint16']
            elif bytesPerVoxel==4:
                bufferType = typeDict['float32']
            elif bytesPerVoxel==8:
                bufferType = typeDict['float64']
            else:
                raise ValueError("File size is not compatible with the given dimensions")
    return bufferType
          
          
def volumeSliceGen(filename, dims, dtype = 'H', headerLength = 0, inds = None):
    """
    Generator that spits out slices of a volume file, if only specified slices [first,last,step]
    are wanted, this can be given as the optional parameter inds. NB: first slice of data is number 0,
    and the last index is excluded like normally in python!
    """
    
    dtype = parseDtype(dtype)
    f = open(filename,'rb')
    
    f.read(headerLength)
    
    if inds is None:
        inds = np.array([0,dims[-1],1])
        
    n=inds[0]
    
    for i in range(0,n):
        dummy = np.fromfile(f,dtype,np.prod(dims[:-1]))
    
    while n<inds[1]:
        #~ keyboard()
        data = np.fromfile(f,dtype,np.prod(dims[:-1]))
        data = np.reshape(data,dims[:-1])
        
        #read in inds[2]-1 additional slices to dummy variable, because 
        #I'm too lazy to figure out how to just move the file position marker
        for i in range(1,inds[2]):
            dummy = np.fromfile(f, dtype, np.prod(dims[:-1]))
        
        n+=inds[2]
        
        yield data
    
    f.close()
    
                                
def readBinaryFile(filename, dims, dtype = 'H', headerLength = 0, indexOrder = 'F'):
    
    dtype = parseDtype(dtype)
    
    f = open(filename, 'rb')
    
    header = f.read(headerLength)
    
    if indexOrder not in ['F','C','A']:
        raise ValueError("Invalid indexing order, should be F,C or A, got %s" % str(indexOrder))
    data = np.fromfile(f,dtype,np.prod(dims))
    data = np.reshape(data,dims,indexOrder)
    
    f.close()
    
    return (header,data)


def readRawVolume(filename,bufferType=None,contiguity='C'):
    
    if not isinstance(filename, str):
        raise TypeError('File name must be a string')

    try:
        fileSize = os.path.getsize(filename)
    except OSError:
        print("Cannot access file {}".format(filename))
        raise

    contiguity = contiguity.upper()
    
    if contiguity not in {'C', 'F'}:
        raise ValueError("Contiguity value must be 'C' or 'F', got {}".format(contiguity))
    
    try:
        if bufferType is None:
            dims,bufferType = getVolumeParamsFromFilename(filename,bufferType)
        else:
            dims = getVolumeParamsFromFilename(fileName)[0]
    except ValueError:
        raise
    
    if bufferType is not None:
        try:
            if type(bufferType) is str:
                bufferType = typeDict[bufferType]
            else:
                typeDict[bufferType]
        except KeyError:
            raise ValueError("Invalid datatype: {}".format(bufferType))
    
    if dims is None:
        raise ValueError("Filename {} cannot be used to deduce volume geometry".format(filename))
    
    if bufferType is None: #Try to guess buffer type based on file size
        numVoxels = np.prod(dims)
        bytesPerVoxel = fileSize//numVoxels
        if fileSize % numVoxels:
            raise ValueError("File size is not compatible with the given dimensions")
        else:
            if bytesPerVoxel==1:
                bufferType = typeDict['uint8']
            elif bytesPerVoxel==2:
                bufferType = typeDict['uint16']
            elif bytesPerVoxel==4:
                bufferType = typeDict['float32']
            else:
                raise ValueError("File size is not compatible with the given dimensions")
    with open(filename,'rb') as f:
        arr = np.fromfile(f,bufferType)
    
    if contiguity == 'C':
        dims = dims[::-1]
        arr.shape = dims
    else:
        arr=np.reshape(arr,dims,order=contiguity)
        
    return arr
    #return np.reshape(arr,dims,contiguity)
        
    
def memMapRawVolume(imgFile,mode='r',bufferType=None):
    '''Memory map raw volume.
    
    If mode is given, then use that, otherwise default to 'r', i.e. read-only. Image datatype can be given as an argument, if not present in the filename or otherwise you want to override it. If none is given and it can't be extracted from the file name, will try to deduce it from the file size and dimensions.
    
    Returns an np.memmap instance representing the raw volume.'''
    
    if bufferType is not None:
        try:
            if type(bufferType) is str:
                bufferType = typeDict[bufferType]
            else:
                typeDict[bufferType]
        except KeyError:
            raise ValueError("Invalid datatype: {}".format(bufferType))
        imgDims = getVolumeParamsFromFilename(imgFile)[0]
    else:
        imgDims,bufferType = getVolumeParamsFromFilename(imgFile)
    
    imgDims = imgDims[::-1]
    
    if bufferType is None:
        bufferType = guessDataType(imgFile,imgDims)
    
    return np.memmap(imgFile,dtype=bufferType,mode=mode,shape=imgDims)
    
    
    
def writeBinaryFile(filename, data, header = None, dtype = None):
    
    if dtype is not None:
        dtype = parseDtype(dtype)
    else:
        dtype = data.dtype.char
    
    f = open(filename,'wb')
    
    if dtype != data.dtype.char:
        data = data.astype(dtype)
    
    if header is not None:
        f.write(header)
    
    data.tofile(f)
    f.close()

    
def saveRawVolumeIJ(img,namestem):
    dims = img.shape
    while len(dims)<3:
        dims = list(dims)
        dims.append(1)
        dims=tuple(dims)
    name = namestem+'_%s_%dx%dx%d.raw' % (typeDict[img.dtype],*dims[::-1])
    with open(name,'wb') as f:
        img.tofile(f)
    return name


def checkType(bufferType):
    try:
        if type(bufferType) is str:
            bufferType = typeDict[bufferType]
        else:
            typeDict[bufferType]
    except KeyError as E:
        raise ValueError('Unknown datatype {}'.format(bufferType)) from E
    
    return bufferType
