'''
Created on Mar 26, 2017

@author: Cheng-lin Li
'''
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

INPUT_FILE = 'classification.txt' #Default input file name
ORIG_STDOUT = None
#OUTPUT_FILE = 'output.txt' # OUTPUT_FILE COULD BE 'OUTPUT_FILE = None' for console or file name (e.g. 'OUTPUT_FILE = 'output.txt') for file.'
OUTPUT_FILE = None # OUTPUT_FILE COULD BE 'OUTPUT_FILE = None' for console or file name (e.g. 'OUTPUT_FILE = 'output.txt') for file.'

def getInputData(filename):
    _data = np.genfromtxt(filename, delimiter = ',')
    _X = _data[1:, 1:12] # variable numbers are 11
    _Y = _data[1:, 12]  # column for label data
    return _X, _Y    

if __name__ == '__main__':

    input_file = ''
    output_file = ''
    
    if len(sys.argv) < 2 : 
        print('Usage of Linear Regression: %s input_matrix.dat output.txt '%(sys.argv[0]))
        print('    input_matrix is the input variable matrix.')
        print('    output.txt will output weights for each dimensions')

    else:
        input_file = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
        output_file = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FILE
        
    
    X, Y = getInputData("summarize-new.csv") #Get column 1,2 as X, column 3 as Z
    
    lr = LinearRegression(normalize=True)
    
    _X = list(X)
     
    lr.fit(X, Y)
    print(str(lr.get_params))
    score = lr.score(X, Y)    
    W = lr.coef_
    print('score =', score)
    print ('W=', W )       

    New_Y = lr.predict(_X)
    print('Prediction:%s'%(New_Y))

          