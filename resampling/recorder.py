import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import plt_utils as pu
import json
import torch

def get_current_date_time():
    now = datetime.now()
    timestamp = now.strftime("%Y-%B-%d %H-%M-%S-%f")
    return timestamp

def log_format(timestamp, line):
    return '{t} {li}'.format(wid=10, pr=0, t=timestamp, li=line)

def summarise_exps(exps_path):
        df = pd.read_csv(exps_path)
        evaluation_metrics = []
        for col in df.columns:
            if col.endswith('_Train') or col.endswith('_Test'):
                em = col.rsplit('_', 1)[0]  # Extract the metric name
                if em not in evaluation_metrics:
                    evaluation_metrics.append(em)
        def string_to_float_list(s):
            s = s.strip('[]')  # Remove square brackets
            float_list = [float(x) for x in s.split()]  # Split and convert to float
            return float_list

        for em in evaluation_metrics:
            df[f'{em}_Train'] = df[f'{em}_Train'].apply(string_to_float_list)
            df[f'{em}_Test'] = df[f'{em}_Test'].apply(string_to_float_list)
            #numpy_array = np.array(df['Column1'].tolist())

        def custom_mean(series):
            return np.mean(series.tolist(), axis=0)

        def custom_std(series):
            return np.std(series.tolist(), axis=0)

        agg_funcs = {
            'Time(s)': ['mean','std']
        }

        for em in evaluation_metrics:
            agg_funcs[f'{em}_Train'] = [custom_mean, custom_std]
            agg_funcs[f'{em}_Test'] = [custom_mean, custom_std]

        df = df.groupby('Res').agg(agg_funcs).reset_index()
        df.rename(columns={"Time(s)": f"Time(s)_ms"}, inplace=True)
        for em in evaluation_metrics:
            df.rename(columns={f'{em}_Train': f"{em}_Train_ms"}, inplace=True)
            df.rename(columns={f'{em}_Test': f"{em}_Test_ms"}, inplace=True)
        
        df1 = pd.DataFrame()
        df1['Res'] = df['Res']
        df1[[f"Time(s)_Mean", f"Time(s)_Std"]] = df["Time(s)_ms"]
        for em in evaluation_metrics:
            df1[[f'{em}_Train_Mean', f'{em}_Train_Std']] = df[f'{em}_Train_ms']
            df1[[f'{em}_Test_Mean', f'{em}_Test_Std']] = df[f'{em}_Test_ms']


        df2 = df1.copy()

        def calculate_mean(lst):
            return sum(lst) / len(lst)

        for em in evaluation_metrics:
            df2[f"{em}_Train_Mean"] = df1[f"{em}_Train_Mean"].apply(calculate_mean)
            df2[f"{em}_Test_Mean"] = df1[f"{em}_Test_Mean"].apply(calculate_mean)
            df2[f"{em}_Train_Std"] = df1[f"{em}_Train_Std"].apply(calculate_mean)
            df2[f"{em}_Test_Std"] = df1[f"{em}_Test_Std"].apply(calculate_mean)
        return df1, df2


class ExperimentRecorder():
    def __init__(self, to_save, data_name, forecasters):
        #dictionary for all save booleans
        self.to_save = to_save
        self.data_name = data_name #used to set session directory
        self.forecasters = forecasters
        self.setup_session_directories()
    
    def saving_any(self):
        #return true if any items in to_save are true
        return any(self.to_save.values())

    #TODO: set save so user can change their mind about saving something mid session
    def set_save(self, option):
        if self.to_save[option]: return
        #TODO: if option not in
        self.to_save[option] = option


    def setup_session_directories(self):
        if not self.saving_any():
            return
        dt = get_current_date_time()
        self.session_path = Path(f"{os.getcwd()}/Sessions/{self.data_name}/{dt}")
        os.makedirs(self.session_path, exist_ok=True)
        
        if self.to_save['forecast_models']:
            self.forecast_models_path = self.session_path.joinpath("forecast_models")
            os.makedirs(self.forecast_models_path, exist_ok=True)
        if self.to_save['forecast_results']:
            self.forecast_results_path = self.session_path.joinpath("forecast_results")
            os.makedirs(self.forecast_results_path, exist_ok=True)
        if self.to_save['resample_models']:
            self.resample_models_path = self.session_path.joinpath("resample_models")
            os.makedirs(self.resample_models_path, exist_ok=True)
        if self.to_save['resample_results']:
            self.resample_results_path = self.session_path.joinpath("resample_results")
            os.makedirs(self.resample_results_path, exist_ok=True)
        if self.to_save['eval_results']:
            self.eval_results_path = self.session_path.joinpath("eval_results")
            os.makedirs(self.eval_results_path, exist_ok=True)
        if self.to_save['figures']:
            self.figures_path = self.session_path.joinpath("figures")
            os.makedirs(self.figures_path, exist_ok=True)
            ffp = {}
            for f in self.forecasters:
                ffp[f] = self.figures_path.joinpath(f)
                os.makedirs(ffp[f], exist_ok=True)
            self.figures_forecasters_path = ffp
            pu.set_save_figs(True, self.figures_path, ffp)
        if self.to_save['parameters']:
            sesh = {}
            #TODO: if the file doesn't already exist
            with open(self.session_path.joinpath("params.json"), 'w') as file:
                json.dump(sesh, file, indent=4)


    def log_print(self,line):
        now = datetime.now()
        timestamp = now.strftime("%H:%M:%S:%f")[:-3]
        log_out = log_format(timestamp, line)
        print(f"LOG: {log_out}\n")
        if not self.to_save['logs']: return
        log_file_path = self.session_path.joinpath("log.txt")
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(log_out+"\n")
            log_file.flush()
            log_file.close()
    
    def save_param_dict(self,pdict):
        if not self.to_save['parameters']: return
        params_path = self.session_path.joinpath("params.json")
        with open(params_path, 'r') as file:
            data = json.load(file)
        data.update(pdict)
        with open(params_path, 'w') as file:
            json.dump(data, file, indent=4)
    
    #should be called on the best GAN, but will work for any
    def save_gan(self, gen, disc, typ, name):
        if not self.to_save['resample_models']: return
        gan_path = self.resample_models_path.joinpath(f"{typ}/{name}")
        os.makedirs(gan_path, exist_ok=True)
        model_gen = torch.jit.script(gen)
        model_gen.save(gan_path.joinpath("generator.pt"))
        model_disc = torch.jit.script(disc)
        model_disc.save(gan_path.joinpath("discriminator.pt"))

    def save_LSTM(self, model, name):
        if not self.to_save['forecast_models']: return
        model.save(self.forecast_models_path.joinpath(f"LSTM_{name}.keras"))        

    def save_resample_results_np(self, results, forecaster):
        if not self.to_save['resample_results']: return    
        self.log_print("Saving resample results numpy style")
        for key, res in results.items():  
            file_name = self.resample_results_path.joinpath(f"{forecaster}_{key}.txt")
            np.savetxt(file_name, res, delimiter=',')

    def save_resample_results_step(self, results, N_STEPS_IN, N_FVARS, FVARS, TVAR):
        if not self.to_save['resample_results']: return
        self.log_print("Saving resample results step style")
        for key, resample in results.items(): 
            file_name = self.resample_results_path.joinpath(f"{key}.csv")
            targets = [arr for arr in resample[:,(N_STEPS_IN*N_FVARS):]]
            features = resample[:,:(N_STEPS_IN*N_FVARS)] 
            features_dict = {FVARS[i] : [arr for arr in features[:,np.arange(i, N_FVARS*N_STEPS_IN+i, N_FVARS)]] for i in range(N_FVARS)}
            save_df = {f"Target_{TVAR}": targets, **features_dict}
            pd.DataFrame(save_df).to_csv(file_name)
    
    #must record results in a format that can be easily loaded to the resampling_results dict
    #need to know how many 
    #def save_sample_results()

    def save_all_evaler_exps(self, df, forecaster):
        if not self.to_save['eval_results']: return
        df.to_csv(self.eval_results_path.joinpath(f"all_{forecaster}_exps.csv"))

    def save_ks(self, ks_results):
        if not self.saving_any(): return
        ks_results.to_csv(self.session_path.joinpath("ks_results.csv"))

    def save_summary_evaler(self, forecaster):
        df1, df2 = summarise_exps(self.eval_results_path.joinpath(f"all_{forecaster}_exps.csv"))
        save_path_1 = self.eval_results_path.joinpath(f"steps_summary_{forecaster}.csv")
        save_path_2 = self.eval_results_path.joinpath(f"full_summary_{forecaster}.csv")
        df1.to_csv(save_path_1)
        df2.to_csv(save_path_2)

'''
#Saving forecast results not finished implementing
if SAVE_FORECAST_RESULTS:
    df_test_data = {}
    df_test_data['Y_test'] = Y_test
    df_test_data['X_test'] = X_test
    for key in forecast_results.keys():
        train_file_name = RESAMPLE_RESULTS_PATH.joinpath(f"{key}_train.csv")
        Y_train_4df = [arr for arr in forecast_results[key]['Y_train']]
        predict_4df = [arr for arr in forecast_results[key]['predict_train']]
        X_train_4df = {f"{FVARS[i]}_X" : [arr for arr in forecast_results[key]['X_train'][:,np.arange(i, N_FVARS*N_STEPS_IN+i, N_FVARS)]] for i in range(N_FVARS)}
        df_train_data = {
            f"predict_{TVAR}":predict_4df,
            f"actual_{TVAR}":Y_train_4df
        }
        df_train_data = {**df_train_data, **X_train_4df}
        print(pd.DataFrame(df_train_data))
        break
        #pd.DataFrame(df_train_data).to_csv(train_file_name)
        df_test_data['predict_test'] = forecast_results[key]['predict_test']
    test_file_name = RESAMPLE_RESULTS_PATH.joinpath(f"{key}_test.csv")
    #pd.DataFrame(df_test_data).to_csv(test_file_name)
    
    
    #targets = [arr for arr in resample[:,(N_STEPS_IN*N_FVARS):]]
    #features = resample[:,:(N_STEPS_IN*N_FVARS)] 
    #features_dict = {FVARS[i] : [arr for arr in features[:,np.arange(i, N_FVARS*N_STEPS_IN+i, N_FVARS)]] for i in range(N_FVARS)}
    #save_df = {f"Target {TVAR}": targets, **features_dict}
#print(EVALER.getExperimentsSummaryTable())
#print(EVALER.getExperimentsSummaryTable(metric="CaseWeight"))
'''