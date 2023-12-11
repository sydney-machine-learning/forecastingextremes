import numpy as np
from scipy.special import factorial
from scipy.interpolate import PchipInterpolator


#Perform PCHIP interpolation on distribution of targets
#Return PchipInterpolator based on summary stats of distribution
#y_train is a 1d numpy array

def PCHIP_interpolator(y_train, percentages, rels):
    #q75, q25 = np.percentile(y_train, [75 ,25])
    #iqr = q75 - q25
    #median = np.median(y_train)
    #value = [median - 1.5* iqr, median, median + 1.5 *iqr]
    #relevances = [1,0,1]
    
    ymin = [y_train.min()]
    ymax = [y_train.max()]
    percentiles = np.percentile(y_train, percentages)
    percentiles = np.append(ymin,percentiles)
    percentiles = np.append(percentiles,ymax) 
    
    rels = np.append(0.0,rels)
    rels = np.append(rels,1.0)
    return PchipInterpolator(percentiles, rels, extrapolate=False)

#RELEVANCE_FUNCTION may store this function
#Uses PchipInterpolator (pchip_func) supplied to find relevance of targets in y
#y can be 2d numpy array (num samples, N_STEPS_OUT) or 1d numpy array (num samples) if N_STEPS_OUT == 1
#'combine' argument used to specify how to combine the relevance scores produced for each of n steps out if N_STEPS_OUT > 1
#returns relevance scores as 1d numpy array (num training samples,) if combine!='none' else 2d array (num samples, N_STEPS_OUT)
def PCHIPRelevance(y, pchip_func, combine='none'):
    if isinstance(y,float):
        rels = pchip_func.__call__(y)
        np.nan_to_num(rels, copy=False, nan=1)
        return rels
    if len(y.shape) == 1: #if n steps out is 1
        rels = pchip_func.__call__(y)
        np.nan_to_num(rels, copy=False, nan=1)
        return rels
    
    rels = np.empty_like(y)
    for i in range(y.shape[1]):
        rels[:,i] = pchip_func.__call__(y[:,i])
        np.nan_to_num(rels[:,i], copy=False, nan=1)
    if combine == 'mean':
        return np.mean(rels,axis=1)
    elif combine == 'max':
        return np.max(rels,axis=1)
    elif combine == 'first':
        return rels[:,0]
    else: #None
        return rels
    
def dpois(x,lamb):
    return ((lamb**x)*np.exp(-lamb))/(factorial(x))

def sigmoid(x):
    return 1/(1+np.exp(-x))

#RELEVANCE_FUNCTION may store this function
def poisRelevance(data, smean, sstd, rshift, combine='none'):
    '''
    Computes a relevance score.
    args:
        data: 1D nump array of target values
        smean: sample mean of training samples
        sstd: sample standard distribution of training samples (might come in handy for normally distributed data)
        rshift: experiment with shifting the relevance function by 2*rshift units to the right
    returns:
        1D numpy array of relevance scores corresponding to samples
    '''
    if isinstance(data,float):
        rels = sigmoid((1/dpois(data,smean))-2*rshift)
        np.nan_to_num(rels, copy=False, nan=1)
        return rels
    if len(data.shape) == 1: #if n steps out is 1
        rels = sigmoid((1/dpois(data,smean))-2*rshift)
        np.nan_to_num(rels, copy=False, nan=1)
        return rels
    
    rels = np.empty_like(data)
    for i in range(data.shape[1]):
        rels[:,i] = sigmoid((1/dpois(data[:,i],smean))-2*rshift)
        np.nan_to_num(rels[:,i], copy=False, nan=1)
    if combine == 'mean':
        return np.mean(rels,axis=1)
    elif combine == 'max':
        return np.max(rels,axis=1)
    elif combine == 'first':
        return rels[:,0]
    else: #None
        return rels
    
#Return approximate inverse of PCHIP function for one relevance score
#Just used to plot target value that intersects with RELEVANCE_THRESHOLD
#Could find any inbuilt inverse for PchipInterpolator so had to go with this silly method
def PCHIPApproxInverse(rscore, rel_func, ymin=0.0, ymax=0.99):
    below, above = ymin, ymax
    guess = np.random.uniform(below,above)
    rguess = rel_func.__call__(guess)
    i = 0
    while abs(rscore - rguess) > 0.01 and i < 10000:
        if rscore - rguess > 0:
            below = guess
        if rscore - rguess < 0:
            above = guess
        guess = np.random.uniform(below,above)
        rguess = rel_func.__call__(guess)
        i+=1
    
    return guess