import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, mean_squared_error, fbeta_score, roc_curve

from pathlib import Path
from tabulate import tabulate

class Evaluator():
    def __init__(self, eval_params, rel_thresh, rel_func):
        self.eval_metrics = []
        self.all_metric_names = []
        for metric in EvaluationMetric.__subclasses__():
            m_name = metric.__name__
            if m_name not in eval_params.keys(): continue
            new_metric = metric(eval_params[m_name], rel_func, rel_thresh)
            self.eval_metrics.append(new_metric)
            self.all_metric_names = self.all_metric_names + new_metric.m_names
        self.results_df = self.initResultsDf()
    
    def initResultsDf(self):
        cols = []
        for metric in self.eval_metrics:
            cols = cols + metric.m_names
        cols = [item + suffix for suffix in ['_Train','_Test'] for item in cols]
        cols = ["Exp","Res","Time(s)"]+cols
        return pd.DataFrame(columns=cols) 
            
    def evaluateMetrics(self, exp, res, train_pred, train_actual, test_pred, test_actual, time):
        exp_row = {"Exp": exp, "Res": res, "Time(s)":time}
        for metric in self.eval_metrics:
            tr = metric.evaluate(train_pred, train_actual)
            tr = {key + "_Train": value for key, value in tr.items()}
            ts = metric.evaluate(test_pred, test_actual)
            ts = {key + "_Test": value for key, value in ts.items()}
            exp_row = {**exp_row, **tr, **ts}
        self.results_df = self.results_df.reset_index(drop=True)
        self.results_df = pd.concat([self.results_df, pd.DataFrame([exp_row])], ignore_index=True)
    
    def getMetricScore(self, exp, res, metric, test=True):
        df_col = metric + ("_Test" if test else "_Train")
        df_expres = self.results_df.loc[(self.results_df['Exp'] == exp) & (self.results_df['Res'] == res)]
        try:
            df_met = df_expres[df_col]
        except KeyError as e:
            print(f"Metric not found: {e}")
            return None
        score = df_met.values[0]
        return score
    
    def getExperimentsSummaryTable(self,metric=None):
        #except KeyError as e:
        if metric is None:
            metrics = self.all_metric_names #[mn for m in self.eval_metrics for mn in m]
        elif metric not in self.all_metric_names or metric == "RelevanceROC":
            print(f"{metric} not found")
            return None
        else:
            metrics = [metric]
        metrics = [s for s in metrics if "RelevanceROC" not in s]
        strat_meta = [['Resampling','Metric','Train_Mean','Test_Mean','Train_Std','Test_Std']]
        for resname in self.results_df['Res'].unique():
            for m in metrics:
                #Means for metrics accross experiments
                tr_m = self.results_df[self.results_df['Res'] == resname][m + "_Train"].mean()
                te_m = self.results_df[self.results_df['Res'] == resname][m + "_Test"].mean()
                #Stds for metrics
                if self.results_df[self.results_df['Res'] == resname]['Exp'].max() >= 2:
                    tr_std = self.results_df[self.results_df['Res'] == resname][m + "_Train"].to_numpy().std()
                    te_std = self.results_df[self.results_df['Res'] == resname][m + "_Test"].to_numpy().std()
                else:
                    tr_std = np.zeros_like(self.results_df[m + "_Train"][0])
                    te_std = np.zeros_like(self.results_df[m + "_Test"][0])
                tab_row = [resname, m, str(tr_m), str(te_m), str(tr_std), str(te_std)]
                strat_meta.append(tab_row)
        return tabulate(strat_meta,headers='firstrow',tablefmt='grid')
    
    def getExperimentSummaryDf(self, metric, test=True):
        mlabel = f"{metric}_{'Test' if test else 'Train'}"
        sum_df = self.results_df[['Exp', 'Res', mlabel]]
        def custom_agg(series):
            mean_vals = np.mean(series.tolist(), axis=0)
            std_vals = np.std(series.tolist(), axis=0)
            return pd.Series([mean_vals, std_vals], index=['Mean', 'Std'])
        result_df = sum_df.groupby('Res')[mlabel].agg(custom_agg).reset_index()
        result_df.rename(columns={mlabel: f"{mlabel}_ms"}, inplace=True)
        result_df[[f"{mlabel}_Mean", f"{mlabel}_Std"]] = pd.DataFrame(result_df[f"{mlabel}_ms"].tolist(), index=result_df.index)
        result_df.drop(columns=[f"{mlabel}_ms"], inplace=True)
        return result_df

    def getResultsDf(self):
        return self.results_df.drop(['RelevanceROC_Train','RelevanceROC_Test'], axis=1, errors='ignore')
        #save_df.to_csv(session_path.joinpath("eval_results.csv"))


def biVariateAutoRelevance(prel, arel, m):
    return np.add((1-m)*prel, m*arel)

class EvaluationMetric:
    def __init__(self):
        pass
    
    def evaluate(self):
        pass

class RMSE(EvaluationMetric):
    m_names = ["RMSE"]
    def __init__(self,params,rel_func,rel_thresh):
        pass

    def evaluate(self, pred, actual):
        return { "RMSE" : np.sqrt(mean_squared_error(actual, pred,multioutput='raw_values'))}
        
    
class RMSERare(EvaluationMetric):
    m_names = ["RMSERare"]
    def __init__(self,params,rel_func,rel_thresh):
        self.rel_func = rel_func
        self.rel_thresh = rel_thresh
        pass

    def evaluate(self, pred, actual):
        actual_rels = self.rel_func.__call__(actual)
        rmse_rares = np.array([])
        for i in range(0,actual_rels.shape[1]):
            rare_ind = np.argwhere(actual_rels[:,i] >= self.rel_thresh).flatten()
            rmse_rare = np.sqrt(mean_squared_error(actual[rare_ind,i], pred[rare_ind,i]))
            rmse_rares = np.append(rmse_rares, rmse_rare)
        '''
        actual_rel = self.rel_func.__call__(actual, combine='first')
        rare_ind = np.argwhere(actual_rel >= self.rel_thresh).flatten()
        rmse_rare = mean_squared_error(actual[rare_ind], pred[rare_ind], multioutput='raw_values')
        '''
        return {"RMSERare": rmse_rares}

#ToDo: this isn't implemented correctly
class Utility(EvaluationMetric):
    m_names = ["Utility","Benefit","Cost"]
    def __init__(self,params,rel_func,rel_thresh):
        self.rel_func = rel_func
        self.rel_thresh = rel_thresh
        #self.bmax = params['B_max']
        #self.cmax = params['C_max']
        self.mbirel = params['m']
    
    def utilityBasedCost(self, pred, actual):
        pred_rel = self.rel_func.__call__(pred)
        actual_rel = self.rel_func.__call__(actual)
        bivrel = biVariateAutoRelevance(pred_rel, actual_rel, self.mbirel)
        loss = np.abs(pred - actual)
        return np.multiply(bivrel, loss)

    def utilityBasedBenefit(self, pred, actual):
        actual_rel = self.rel_func.__call__(actual)
        loss = np.abs(pred - actual)
        return np.multiply(actual_rel, 1-loss)

    def evaluate(self, pred, actual):
        benefit = np.sum(self.utilityBasedBenefit(pred, actual),axis=0) / pred.shape[0]
        cost = np.sum(self.utilityBasedCost(pred, actual),axis=0) / pred.shape[0]
        utility = benefit - cost
        return {"Utility": utility, "Benefit": benefit, "Cost": cost}

class CaseWeight(EvaluationMetric):
    m_names = ["CaseWeight"]
    def __init__(self,params,rel_func,rel_thresh):
        self.rel_func = rel_func
        self.rel_thresh = rel_thresh
        self.mbirel = params['m']
    
    def evaluate(self, pred, actual):
        #pred = np.reshape(pred,(-1,))
        pred_rel = self.rel_func.__call__(pred, combine='none')
        actual_rel = self.rel_func.__call__(actual, combine='none')
        biRel = biVariateAutoRelevance(pred_rel,actual_rel,self.mbirel)
        loss = (pred-actual)**2

        caseMult = np.multiply(biRel,loss)
        case_weight = np.sum(caseMult,axis=0) / np.sum(biRel,axis=0)
        return {"CaseWeight": case_weight}

class PrecisionRecall(EvaluationMetric):
    m_names = ["Precision", "Recall"]
    def __init__(self,params,rel_func,rel_thresh):
        self.rel_func = rel_func
        self.rel_thresh = rel_thresh
        
    def evaluate(self, pred, actual):
        ind_pred = self.rel_func.__call__(pred,combine='first')
        ind_actual = self.rel_func.__call__(actual,combine='first')
        ind_actual[ind_actual <= self.rel_thresh] = 0
        ind_actual[ind_actual > self.rel_thresh] = 1
        ind_pred[ind_pred <= self.rel_thresh] = 0
        ind_pred[ind_pred > self.rel_thresh] = 1
        precision = precision_score(ind_actual, ind_pred, average=None) #how does this work accross axis 1??
        recall = recall_score(ind_actual, ind_pred, average=None)
        return {"Precision": np.array(precision), "Recall": np.array(recall)}
    

class FScore(EvaluationMetric):
    m_names = ["FScore"]
    def __init__(self,params,rel_func,rel_thresh):
        self.rel_func = rel_func
        self.rel_thresh = rel_thresh
        self.beta = params['beta']
        
    def evaluate(self, pred, actual):
        ind_pred = self.rel_func.__call__(pred,combine='none') #return 2d array, rel scores for each step out
        ind_actual = self.rel_func.__call__(actual,combine='none')
        ind_actual[ind_actual <= self.rel_thresh] = 0
        ind_actual[ind_actual > self.rel_thresh] = 1
        ind_pred[ind_pred <= self.rel_thresh] = 0
        ind_pred[ind_pred > self.rel_thresh] = 1
        f_scores = list()
        for i in range(ind_pred.shape[1]):
            f_scores.append(fbeta_score(ind_actual[:,i], ind_pred[:,i], beta=self.beta))
        return {"FScore": np.array(f_scores)}

class RelevanceROC(EvaluationMetric):
    m_names = ["RelevanceROC"]
    def __init__(self,params,rel_func,rel_thresh):
        self.rel_func = rel_func
        self.rel_thresh = rel_thresh
        
    def evaluate(self, pred, actual):
        ind_pred = self.rel_func.__call__(pred,combine='none') #return 2d array, rel scores for each step out
        ind_actual = self.rel_func.__call__(actual,combine='none')
        ind_actual[ind_actual <= self.rel_thresh] = 0
        ind_actual[ind_actual > self.rel_thresh] = 1
        rocs = list()
        for i in range(ind_pred.shape[1]):
            rocs.append(roc_curve(ind_actual[:,i], ind_pred[:,i]))
        return {"RelevanceROC": rocs}



#ev = Evaluator(2,eval_params, relevance_function, RELEVANCE_THRESH)
#for ex in range(2):
#    ev.evaluateMetrics(ex,1,2,3,4)
#print(ev.results_df)