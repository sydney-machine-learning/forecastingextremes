import numpy as np
import pandas as pd
import math
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

def regularResample(x, y, num_rel, num_com, k, under=False):
    '''
    Oversampling using a regular SMOTER approach. 
    Args:
        x: 2D numpy array of time series training samples
        y: 1D numpy array of 0,1s --> 0 if corresponding target value is common, 1 if relevant
        num_rel: number of relevant cases we want after oversampling
        under: Can activate random under sampling by setting under=True
    Returns:
        X_res: 2D numpy array of resampled time series training samples
        y_res: 1D numpy array of 0,1s for resampled data --> 0 if corresponding target value is common, 1 if relevant
    '''
    #sm = SMOTE(sampling_strategy=ratio, random_state=2, k_neighbors=3)
    sm = SMOTE(sampling_strategy={1: num_rel}, random_state=2, k_neighbors=k)
    steps = [('over', sm)]
    if under:
        ru = RandomUnderSampler(random_state=2)
        steps = steps+[('under', ru)]
    pipeline = Pipeline(steps=steps)
    X_res, y_res = pipeline.fit_resample(x, y.ravel())
    return X_res, y_res


def getBinDataFrame(x, rel, rel_scores):
    '''
    Helper function for relBinResample. Returns a dataframe containing information about relevance score, relevance bin number and bin size
    Args:
        x: 2D numpy array of time series training samples
        rel: 1D numpy array of 0,1s --> 0 if corresponding target value is common, 1 if relevant
        rel_scores: 1D numpy array of relevance scores for target value embedded as last element of each training sample in x
    Returns:
        bDf: Pandas data frame containing the following columns:
            'sample': Taken embedded time series data sample
            'bin_num': id for the relevance bin of the sample
            'rel_score': relevance score for the sample
            'relevance': 1 if the sample is relevant, 0 if common
            'bin_size': number of samples in bin_num
    ''' 
    bin_nums = np.array([])
    bn = 1
    prev_rb = rel[0]
    for r in rel:
        if r != prev_rb:
            bn+=1
        bin_nums = np.append(bin_nums, bn)
        prev_rb = r
    bDf = pd.DataFrame({"sample": list(x), "rel_score": rel_scores, "relevance": rel, "bin_num": bin_nums})
    #gen column for number of samples in a rel bin
    bc = bDf.groupby('bin_num')['bin_num'].count()
    bc.name = 'bin_size'
    bDf = bDf.join(bc,on='bin_num')
    return bDf


def getBiasPreferences(nearest, rel_scores, rel_bias, temp_bias):
    '''
        Returns a list of preference scores for k nearest neighbors based on relevance and temporal bias
        Preference scores are used in chooseNeighbors to randomly choose one of k nearest neighbors for interpolation. 
        Args:
            nearest: 1D numpy array of ints representing the indicies of the nearest k neighbors
            rel_scores: 1D numpy array of floats representing relevance socres of the k nearest neighbors
        Returns:
            ps: 1D list of preference scores btw [0,1], sum(ps)==1
    '''
    ps = [1/nearest.size for _ in nearest] #start with uniform prob of choosing each neighbor
    if temp_bias: #assign more preference to the most recent samples in the bin, while maintaining sum(ps)==1
        time = nearest + 1 #represents time
        ps = [(t*p)/sum(np.multiply(time,ps)) for t,p in zip(time,ps)] 
    if rel_bias: #assign more preference to higher relevance scores, while maintaining sum(ps)==1
        #relsp = [relScores[n] for n in nearest] #rel scores corresponding to index in nearest
        
        min_rel = min(rel_scores)
        dot_mults = [(rel_scores[n]-min_rel)*100 + 1 for n in nearest]
        relsp = [dm/sum(dot_mults) for dm in dot_mults]
        
        ps = [(r*p)/sum(np.multiply(relsp,ps)) for r,p in zip(relsp,ps)]
    return ps


def chooseNeighbors(seed, rbin, rels, nn, k, relBias, tempBias):
    '''
        Given a relevance bin and seed sample, returns a list of nn training samples to be used for SMOTER interpolation in genSynthCases
        
        Args:
            seed: 1D numpy array containing the sample used as seed for SMOTER interpolation
            rbin: 2D numpy array of samples in the relevance bin which seed belongs to
            rels: 1D array of relevance scores for samples in rbin
            nn: int, number of times we choose a neighbor (number of synthetic cases we want to generate per seed)
            k: int, number of nearest neighbors to consider
        Returns:
            synth_ns: 2D numpy array of samples from rbin to be used to generate nn new synthetic cases in genSynthCases
    '''
    if k+1 > rbin[:,-1].size: #truncate k if not enough samples in rbin
        k = rbin[:,-1].size - 1
    
    knn = KNeighborsRegressor(n_neighbors = k+1) # k+1 because the case is the nearest neighbor to itself
    knn.fit(rbin, rbin)
    neighbors = knn.kneighbors([seed,seed], return_distance=False)
    nearest = neighbors[1:][0][1:]
    
    #randomly choose nn nearest neighbors
    synth_ns = np.empty([nn, len(rbin[0])])
    for i in range(0,nn):
        ps = getBiasPreferences(nearest, rels, relBias, tempBias)
        synth_n = np.random.choice(nearest,size=1,p=ps)
        synth_ns[i] = rbin[synth_n][0]
    return synth_ns

def genSynthCases(df, nn, k, relBias, tempBias):
    '''
    Performs the SMOTER algorithm on embedded time series data, returns new synthetic cases
    Args:
        df: pandas data frame containing candidate relevant samples with columns:
            'sample': Taken embedded time series data sample
            'bin_num': id for the relevance bin of the sample
            'rel_score': relevance score for the sample
            'relevance': 1 if the sample is relevant, 0 if common
            'bin_size': number of samples in bin_num
            'nsynths': number of new samples to generate using current samples as seed
        nn: number of synthetic samples to generate per sample in df
        k: number of neighbors to consider in SMOTER algorithm
    Returns:
        synth_cases: 2D array of new synthetic cases
    '''
    synth_cases = np.empty((0,len(df['sample'].iloc[0])), float)
    for index, case in df.iterrows():
        #get all other samples in that bin
        binFor = df[df['bin_num'] == case['bin_num']]
        nn =  case['nsynths'] #comment this line out to return to ye old ways of smotering
        seed = case['sample']
        synth_ns = chooseNeighbors(seed, np.stack(binFor['sample'].values, axis=0), binFor['rel_score'].values, nn=nn, k=k, relBias = relBias, tempBias = tempBias)
        #interpolate synthetic cases
        for rep in synth_ns:
            synth_attr = []
            for s, n in zip(seed, rep):
                diff = s - n
                rand_num = np.random.uniform(0,1,1)[0]
                sa = n + rand_num*diff
                synth_attr = synth_attr + [sa]
            synth_cases = np.append(synth_cases, np.array([synth_attr]), axis=0)
    return synth_cases

def relBinResample(x, y, nr, rel_y, k, temp_bias=False, rel_bias=False):
    '''
    args:
        x: 2D numpy array of time series training samples
        y: 1D numpy array of 0,1s indicating a relevant/common sample
        nr: number of relevant samples to generate
        rel_y: 1D numpy array of relevance scores for target value embedded as last element of each training sample in x
        under: activates undersampling (not currently implemented)
        temp_bias: activates temporal bias
        rel_bias: activates relevance bias
    returns:
        x_res:
    SMOTER variations:
        SM_B: temp_bias=False and rel_bias=False
        SM_T: temp_bias=True and rel_bias=False
        SM_Phi: temp_bias=False and rel_bias=True
        SM_TPhi: temp_bias=True and rel_bias=True
    '''
    #get special dataframe providing info on relevance bins
    bDf = getBinDataFrame(x,y,rel_y)
    #get relevant samples where bin size > 1
    smoter_candidates = bDf[(bDf['bin_size'] > 1) & (bDf['relevance'] == 1)].drop("relevance",axis=1)
    rus_candidates = bDf[bDf['relevance'] == 0].drop("relevance",axis=1)
    #bin_weights = smoter_candidates['bin_size'].values / smoter_candidates.groupby('bin_num')['bin_size'].mean().sum()
    #bin_weights = smoter_candidates['bin_size'].values / len(smoter_candidates)
    #smoter_candidates['nsynths'] = (bin_weights * nr).astype(int)
    #print(smoter_candidates[['bin_num','bin_size','nsynths']].to_string(header=True, index=True))
    #print(f"Sum of nsynths: {smoter_candidates['nsynths'].values.sum()}")
    #print(smoter_candidates.drop(['sample'],axis=1))
    #nsynthdf = smoter_candidates.groupby('bin_num')[['nsynths', 'bin_size']].mean()
    #print(nsynthdf.to_string(header=True, index=True))
    #print(f"Total news: {smoter_candidates['nsynths'].sum()}")
    #for _, row in nsynthdf.iterrows():
    #    print(row.head())
    
    synths_per_bin = dict()

    # Calculate the number of synthetic samples per bin based on proportion to 'bin_size'
    for bin_num, group in smoter_candidates.groupby('bin_num'):
        bin_size = group['bin_size'].iloc[0]
        synths_per_bin[bin_num] = (nr * bin_size) // len(smoter_candidates.index)
    smoter_candidates['nsynths'] = 0
    for bn in smoter_candidates['bin_num'].unique():
        to_distribute = synths_per_bin[bn]
        bin_size = smoter_candidates[smoter_candidates['bin_num']==bn]['bin_size'].iloc[0]
        i = 0
        #print(f"bin num: {bn} has {bin_size} and we're distrbuting {to_distribute}")
        while to_distribute != 0:
            if i == bin_size: i = 0
            row_index = smoter_candidates[smoter_candidates['bin_num'] == bn].index[i]
            smoter_candidates.at[row_index, 'nsynths'] += 1
            #curr = smoter_candidates[smoter_candidates['bin_num']==bn]['nsynths'].iloc[i]
            #smoter_candidates[smoter_candidates['bin_num']==bn]['nsynths'].iloc[i] = curr + 1
            
            to_distribute-=1
            i+=1
    #print(f"num to gen: {smoter_candidates['nsynths'].sum()}")
    
    #print(smoter_candidates[['bin_num','bin_size','nsynths']].to_string(header=True, index=True))
    num_sc = len(smoter_candidates.index)
    num_rc = len(rus_candidates.index)
    num_iso = len(bDf.index) - num_sc - num_rc
    #print(f"Num smoter_candidates: {num_sc}")
    #print(f"Number of coms: {num_rc}")
    #print(f"Num isos: {num_iso}")
    synth_mult = (num_rc - num_iso)/num_sc
    #SMOTER these binNums
    rs = genSynthCases(smoter_candidates, synth_mult, k, rel_bias, temp_bias)
    #print(f"Actual number of news (rs.shape): {rs.shape}")
    #print(f"Shape of orig set x: {x.shape}")
    x_res = np.append(x, rs, axis=0)
    #print(f"Returning x_res of shape: {x_res.shape}")
    return x_res


def countRelVsCom(xs, relevance_function, relevance_threshold, combine):
    tot = xs.shape[0]
    tot_rel = (relevance_function.__call__(xs, combine=combine) >= relevance_threshold).sum()
    tot_com = (relevance_function.__call__(xs, combine=combine) < relevance_threshold).sum()
    return tot, tot_rel, tot_com