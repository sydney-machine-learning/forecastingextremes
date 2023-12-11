import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

sns.set_style("darkgrid", {"grid.color": "white", "axes.edgecolor": "black"})
sns.set(rc={"axes.edgecolor": "black","axes.linewidth": 0.9})
sns.set_palette("viridis")
sns.set_theme(style='white')
strat_cols = {
    'no_resample' : '#007ACC', #(Blue)
    'GAN_FNN' : '#DC143C', #(Red)
    '1D-GAN' : '#DC143C', #(Red)
    #'GAN_CNN' : '#92D400', #(Green)
    'GAN_CNN' : '#669400', #(Green)
    '1D-Conv GAN' : '#669400', #(Green)
    'SMOTER_bin_r' : '#b28900', #(Yellow)
    'SMOTER_regular' : '#FF6F61', #(Orange)
    'SMOTER_bin_t' : '#A200FF', #(Purple)
    'SMOTER_bin' : '#00D6E3', #(Cyan)
    'SMOTER_bin_tr' : '#FF00A6', #(Magenta)
    'relevance' : '#8D93AB', #(Grey)
    'No-resampling' : '#8D93AB', #(Grey),
    'SMOTE-R-bin-t': '#8D93AB', #(Grey)
    'SMOTE-R-bin-r': '#8D93AB', #(Grey)
    'SMOTE-R-bin': '#8D93AB', #(Grey)
    'SMOTE-R-bin-tr': '#8D93AB', #(Grey)
    'SMOTE-R': '#8D93AB', #(Grey)
}

#FFD700 (Gold)
#FF00A6 (Magenta)

#factor < 1.0 makes the color darker (e.g., 0.5 for 50% darker)
#factor > 1.0 makes the color brighter (e.g., 1.5 for 50% brighter)
def adjust_brightness(hex_color, factor):
    # Convert hex color to RGB
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    # Adjust brightness
    adjusted_rgb = tuple(min(max(int(value * factor), 0), 255) for value in rgb)
    # Convert RGB back to hex
    adjusted_hex_color = '#{:02x}{:02x}{:02x}'.format(*adjusted_rgb)
    
    return adjusted_hex_color


def rohit(input_string):
    # Replace underscores with spaces
    processed_string = input_string
    if processed_string == 'no_resample': processed_string = 'No Resampling'
    if processed_string == 'SMOTER_bin': processed_string = 'SMOTE-R Bin'
    if processed_string == 'SMOTER_bin_r': processed_string = 'SMOTE-R Bin-R'
    if processed_string == 'SMOTER_bin_t': processed_string = 'SMOTE-R Bin-T'
    if processed_string == 'SMOTER_bin_tr': processed_string = 'SMOTE-R Bin-TR'
    if processed_string == 'SMOTER_regular': processed_string = 'SMOTE-R Regular'
    processed_string = processed_string.replace('_', ' ')
    return processed_string


SAVE_FIGS = False
FIGS_PATH = None
FIGS_FORE_PATHS = {}

def set_save_figs(sf, fp, ffp):
    global SAVE_FIGS, FIGS_PATH, FIGS_FORE_PATHS
    SAVE_FIGS = sf
    FIGS_PATH = fp
    FIGS_FORE_PATHS = ffp


def get_save_figs():
    return SAVE_FIGS

def get_figs_path():
    return FIGS_PATH

def get_figs_forecasters_paths():
    return FIGS_FORE_PATHS

def PlotScaledData(data, tvar):
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(131)
    ax_hist = fig.add_subplot(132)
    ax_box = fig.add_subplot(133)

    ax.plot(data)
    ax.set_title('Scaled Data')

    ax_hist.hist(data, bins=20)
    ax_hist.set_title('Scaled Distribution')

    ax_box.boxplot(data, vert=False)
    #ax_box.set_title('Scaled Boxplot')
    if SAVE_FIGS:
        fig.savefig(FIGS_PATH.joinpath(f"scaled_for_{tvar}.png"))

    plt.tight_layout()
    plt.show()

def PlotRelevance(data, rel, rel_thresh, thresh_inv, tvar, dname):
    tick_positions = np.arange(0, 1.1, 0.1)
    
    fig = plt.figure(figsize=(15, 5))
    ax_hist = fig.add_subplot(121)
    ax_rel = fig.add_subplot(122)

    #ax_hist.hist(data,bins=70)
    sns.histplot(data=data, bins=70, ax=ax_hist, color=strat_cols['no_resample'])
    ax_hist.axvline(x=thresh_inv, color='k', linestyle=':', label=f"Extreme Threshold: {round(thresh_inv,3)}")
    #ax_hist.set_title(f"{dname} Disribution", fontsize=14)
    ax_hist.set_xlabel(tvar, fontsize=10)
    ax_hist.set_ylabel("Frequency", fontsize=10)
    #ax_hist.text(thresh_inv + 0.01, ax_hist.get_ylim()[1] - 200, f"Extreme Threshold: {round(thresh_inv,3)}", color='k')
    ax_hist.set_xticks(tick_positions)
    ax_hist.legend(shadow=True)
    
    figb = plt.figure(figsize=(10, 3))
    ax_box = figb.add_subplot(111)
    sns.boxplot(data=data, orient='h', ax=ax_box, fliersize=1, color=strat_cols['no_resample'])
    ax_box.axvline(x=thresh_inv, color='k', linestyle=':', label=f"Extreme Threshold: {round(thresh_inv,3)}")
    #ax_box.set_title(f"{dname} Boxplot", fontsize=14)
    ax_box.set_xlabel(tvar, fontsize=10)
    ax_box.set_xticks(tick_positions)
    ax_box.legend(fontsize=10, loc='upper right', shadow=True)
    
    #sns.plot(x=data, y=rel, s=2 , ax=ax_rel, label="Relevance", color=strat_cols['relevance'])
    ax_rel.plot(data, rel, marker='.', linestyle='none', markersize=0.5, label="Relevance", color=strat_cols['relevance'])
    ax_rel.axhline(y=rel_thresh, color='k', linestyle='--', label=f"Relevance Threshold: {rel_thresh}")
    ax_rel.axvline(x=thresh_inv, color='k', linestyle=':', label=f"Extreme Threshold: {round(thresh_inv,3)}")
    #ax_rel.set_title(f"{dname} Relevance Function", fontsize=14)
    ax_rel.set_xlabel(tvar, fontsize=10)
    ax_rel.set_ylabel("Relevance Score", fontsize=10)
    ax_rel.set_xticks(tick_positions)
    ax_rel.set_yticks(tick_positions)
    legend = ax_rel.legend(shadow=True)
    for legend_handle in legend.legendHandles:
        legend_handle.set_markersize(10)
    

    if SAVE_FIGS:
        fig.savefig(FIGS_PATH.joinpath(f"relevance_for_{tvar}.png"))
        figb.savefig(FIGS_PATH.joinpath(f"boxplot_for_{tvar}.png"))

    plt.tight_layout()
    plt.show()

def biRel(prel, arel, m):
    return np.add((1-m)*prel, m*arel)

def PlotCaseWeightSurface(targets, rel_func, thresh, m, tvar):
    train_target_range = np.linspace(min(targets),max(targets),100)
    train_predicted_range = train_target_range
    Y_true_grid, Y_predicted_grid = np.meshgrid(train_target_range, train_predicted_range)

    def CW(pred, actual, rel_func, m):
        pred_rel = rel_func.__call__(pred, combine='none')
        actual_rel = rel_func.__call__(actual, combine='none')
        br = biRel(pred_rel,actual_rel,m)
        loss = (pred-actual)**2
        caseMult = np.multiply(br,loss)
        return caseMult

    loss_grid = np.vectorize(CW)(Y_true_grid, Y_predicted_grid, rel_func, m)
    fig = plt.figure(figsize=(15,5))
    ax_2d = fig.add_subplot(121)
    ax_3d = fig.add_subplot(122,projection='3d')
    axim = ax_2d.imshow(loss_grid, extent=[min(targets), max(targets), min(targets), max(targets)],origin='lower', aspect='auto', cmap='viridis')
    ax_2d.axhline(y=thresh, color='k', linestyle=':', label='Thresh')
    ax_2d.axvline(x=thresh, color='k', linestyle=':', label='Thresh')
    ax_2d.set_xlabel(f"y ({tvar})")
    yhat = "y\u0302"
    ax_2d.set_ylabel(f"{yhat} ({tvar})")
    ax_2d.set_title(f'Case Weight Surface for {tvar}')
    ax_2d.legend([f"Extreme Threshold: {round(thresh,3)}"])
    ax_3d.plot_surface(Y_true_grid, Y_predicted_grid, loss_grid, cmap='viridis')
    ax_3d.set_xlabel(f"y ({tvar})")
    ax_3d.set_ylabel(f"{yhat} ({tvar})")
    ax_3d.set_zlabel('Case Weight')
    ax_3d.set_title(f'3D Case Weight Surface for {tvar}')
    ax_3d.view_init(elev=30, azim=-110)
    fig.colorbar(mappable=axim,label='Case Weight',ax=ax_2d)

    if SAVE_FIGS:
        fig.savefig(FIGS_PATH.joinpath(f"surface_case_weight_for_{tvar}.png"))

    plt.tight_layout()
    plt.show()


def PlotUtilitySurface(targets, rel_func, thresh, m, tvar):
    train_target_range = np.linspace(min(targets),max(targets),100)
    train_predicted_range = train_target_range
    Y_true_grid, Y_predicted_grid = np.meshgrid(train_target_range, train_predicted_range)


    def uBC(pred,actual,rel_func,m=0.5):
        pred_rel = rel_func.__call__(pred)
        actual_rel = rel_func.__call__(actual)
        br = biRel(pred_rel, actual_rel, m)
        loss = np.abs(pred - actual)
        return np.multiply(br, loss)

    def uBB(pred, actual, rel_func):
        actual_rel = rel_func.__call__(actual)
        loss = np.abs(pred - actual)
        return np.multiply(actual_rel, 1-loss)

    def util(pred,actual,rel_func,m):
        benefit = np.sum(uBB(pred, actual,rel_func))
        cost = np.sum(uBC(pred, actual,rel_func,m))
        utility = benefit - cost
        return utility

    loss_grid = np.vectorize(util)(Y_true_grid, Y_predicted_grid, rel_func, m)
    fig = plt.figure(figsize=(15,5))
    ax_2d = fig.add_subplot(121)
    ax_3d = fig.add_subplot(122,projection='3d')
    axim = ax_2d.imshow(loss_grid, extent=[min(targets), max(targets), min(targets), max(targets)],origin='lower', aspect='auto', cmap='viridis')
    ax_2d.axhline(y=thresh, color='k', linestyle=':', label='Thresh')
    ax_2d.axvline(x=thresh, color='k', linestyle=':', label='Thresh')
    ax_2d.set_xlabel(f"Y ({tvar})")
    yhat = "Y\u0302"
    ax_2d.set_ylabel(f"{yhat} ({tvar})")
    ax_2d.set_title(f'Utility Surface for {tvar}')
    ax_2d.legend([f"Extreme Threshold: {round(thresh,3)}"])
    ax_3d.plot_surface(Y_true_grid, Y_predicted_grid, loss_grid, cmap='viridis')
    
    ax_3d.set_xlabel(f"Y ({tvar})")
    ax_3d.set_ylabel(f"{yhat} ({tvar})")
    ax_3d.set_zlabel('Utility')
    ax_3d.set_title(f'3D Utility Surface for {tvar}')
    ax_3d.view_init(elev=30, azim=-110)
    fig.colorbar(mappable=axim,label='Utility',ax=ax_2d)
    
    if SAVE_FIGS:
        fig.savefig(FIGS_PATH.joinpath(f"surface_utility_for_{tvar}..png"))
    
    plt.tight_layout()
    plt.show()

def PlotBins(data_y, rel_y, thresh, tvar, dname):
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    
    com_col = 'white'
    rel_col = 'white'
    ax.set_facecolor(com_col)
    ax.grid(False)
    for i, val in enumerate(rel_y):
        if val == 1: ax.axvspan(i - 0.5, i + 0.5, color=rel_col)
    sns.lineplot(data=data_y, ax=ax, color=strat_cols['no_resample'], linewidth=0.9, label=tvar)
    #ax.axhline(y=thresh, color='k', linestyle=':', label='Thresh', linewidth=1.0)
    
    ax.set_xlabel("Time (samples)", fontsize=12)
    ax.set_ylabel(tvar, fontsize=12)
    ax.set_title(f"{dname} Relevance Bins", fontsize=16)
    tick_positions = np.arange(0, 1.1, 0.1)
    ax.set_yticks(tick_positions)

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=rel_col, lw=4), Line2D([0], [0], color='k', lw=1, linestyle=':'), Line2D([0], [0], color=strat_cols['no_resample'], lw=1)]
    #ax.legend(custom_lines, ['Relevance Bin', f'Extremes Threshold: {round(thresh,3)}', tvar],fontsize=16, loc='upper right', shadow=True)
    if SAVE_FIGS:
        fig.savefig(FIGS_PATH.joinpath(f"relevance_bins_for_{tvar}.png"))
    
    plt.tight_layout()
    plt.show()

def PlotKDECompare(orig_ex, resamp_ex, rel_func, rel_thresh, step_out, tind, ex_thresh, tvar, dname, forecaster):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    y_orig_rel = rel_func.__call__(orig_ex[:,step_out])
    rare_mask = [val >= rel_thresh for val in y_orig_rel]
    y_orig_rare = orig_ex[rare_mask]
    sns.kdeplot(y_orig_rare[:,step_out], color=strat_cols['no_resample'], label="No Resampling", ax=ax, lw=3, linestyle='dotted')

    for key, res in resamp_ex.items():
        y_res = res[:,tind:]
        y_rel = rel_func.__call__(y_res[:,step_out])
        rare_mask = [val >= rel_thresh for val in y_rel]
        y_rare = y_res[rare_mask]
        if "GAN_CNN" in key: col_key = "GAN_CNN"
        elif "GAN_FNN" in key: col_key = "GAN_FNN"
        else: col_key = key
        sns.kdeplot(y_rare[:,step_out], color=strat_cols[col_key], label=rohit(key), ax=ax, lw=1, alpha=0.9)# linestyle='dashed')

    ax.legend(fontsize=14, loc='upper right', shadow=True)
    ax.set_title(f"{dname} Resampled Extremes Kernel Density Estimates (Normalized)", fontsize=14)
    ax.set_xlabel(tvar, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    tick_positions = np.arange(round(ex_thresh,2), 1.1, 0.1)   
    ax.set_xticks(tick_positions)
    ax.set_xlim(round(ex_thresh, 2)-0.05, 1.05)
    if SAVE_FIGS:
        fig.savefig(FIGS_FORE_PATHS[forecaster].joinpath(f"{forecaster}_kde_comparison.png"))

    plt.tight_layout()    
    plt.show()

def PlotCDFCompare(orig_ex, resamp_ex, rel_func, rel_thresh, step_out, tind, ex_thresh, tvar, dname, forecaster):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    y_orig_rel = rel_func.__call__(orig_ex[:,step_out])
    rare_mask = [val >= rel_thresh for val in y_orig_rel]
    y_orig_rare = orig_ex[rare_mask]

    orig_ex_sorted = np.sort(y_orig_rare[:,step_out])
    cprob_orig = np.arange(1, len(orig_ex_sorted) + 1) / len(orig_ex_sorted)
    for key, res in resamp_ex.items():
        y_res = res[:,tind:]
        y_rel = rel_func.__call__(y_res[:,step_out])
        rare_mask = [val >= rel_thresh for val in y_rel]
        y_rare = y_res[rare_mask]
        resamp_ex_sorted = np.sort(y_rare[:,step_out])
        cprob = np.arange(1, len(resamp_ex_sorted) + 1) / len(resamp_ex_sorted)
        if "GAN_CNN" in key: col_key = "GAN_CNN"
        elif "GAN_FNN" in key: col_key = "GAN_FNN"
        else: col_key = key
        ax.plot(resamp_ex_sorted, cprob, marker='.', linestyle='none', markersize=0.2, label=rohit(key), color=strat_cols[col_key], alpha=0.6)
    ax.plot(orig_ex_sorted, cprob_orig, marker='.', linestyle='none', markersize=2.0, label="No Resampling", color=strat_cols['no_resample'], alpha=0.6)
    
    legend = ax.legend(fontsize=14, loc='lower right', shadow=True)
    for legend_handle in legend.legendHandles:
        legend_handle.set_markersize(10)
    ax.set_title(f"{dname} Resampled Extremes Empirical CDF", fontsize=14)
    ax.set_xlabel(tvar, fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    tick_positions = np.arange(round(ex_thresh,2), 1.1, 0.1)   
    ax.set_xticks(tick_positions)
    ax.set_xlim(round(ex_thresh, 2)-0.05, 1.05)
    
    if SAVE_FIGS:
        fig.savefig(FIGS_FORE_PATHS[forecaster].joinpath(f"{forecaster}_cdf_comparison.png"))
    plt.tight_layout()    
    plt.show()
        
def PlotExremesDistribution(orig_ex, resamp_ex, res_name, step_out, thresh, tvar, dname):
    xh_tick_positions = np.arange(round(thresh,2), 1.1, 0.1)   
    tick_positions = np.arange(0.0, 1.1, 0.1)   

    if "GAN_CNN" in res_name: col_key = "GAN_CNN"
    elif "GAN_FNN" in res_name: col_key = "GAN_FNN"
    else: col_key = res_name
    
    fig = plt.figure(figsize=(20,5))
    ax_h = fig.add_subplot(121)
    ax_cdf = fig.add_subplot(122)
    res_h_col = adjust_brightness(strat_cols[col_key], 1.1)
    no_h_col = adjust_brightness(strat_cols['no_resample'], 1.1)
    ax_h.axvline(x=thresh, color='k', linestyle=':', label=f"Extreme Threshold: {round(thresh,3)}")
    sns.histplot(orig_ex[:,step_out], bins=100, color=no_h_col, ax=ax_h, alpha=0.4, stat='density')
    sns.histplot(resamp_ex[:,step_out], bins=100, color=res_h_col, ax=ax_h, alpha=0.5, stat='density')
    sns.kdeplot(orig_ex[:,step_out], color='k', ax=ax_h, lw=2.5)
    sns.kdeplot(resamp_ex[:,step_out], color='k', ax=ax_h, lw=2.5)
    sns.kdeplot(orig_ex[:,step_out], color=strat_cols['no_resample'], label="No Resampling", ax=ax_h, lw=2)
    sns.kdeplot(resamp_ex[:,step_out], color=strat_cols[col_key], label=rohit(res_name), ax=ax_h, lw=2)
    
    ax_h.legend(fontsize=14, loc='upper right', shadow=True)
    ax_h.set_title(f"{dname} Resampled Extremes Distribution (Normalized)", fontsize=14)
    ax_h.set_xlabel(tvar, fontsize=12)
    ax_h.set_ylabel("Frequency", fontsize=12)
    ax_h.set_xticks(xh_tick_positions)
    ax_h.set_xlim(round(thresh, 2)-0.05, 1.05)

    orig_ex_sorted = np.sort(orig_ex[:,step_out])
    cprob_orig = np.arange(1, len(orig_ex_sorted) + 1) / len(orig_ex_sorted)
    resamp_ex_sorted = np.sort(resamp_ex[:,step_out])
    cprob = np.arange(1, len(resamp_ex_sorted) + 1) / len(resamp_ex_sorted)
    ax_cdf.plot(orig_ex_sorted, cprob_orig, marker='.', linestyle='none', markersize=2, label="No Resampling", color=strat_cols['no_resample'])
    ax_cdf.plot(resamp_ex_sorted, cprob, marker='.', linestyle='none', markersize=0.5, label=rohit(res_name), color=strat_cols[col_key])
    ax_cdf.axvline(x=thresh, color='k', linestyle=':', label=f"Extreme Threshold: {round(thresh,3)}")
    ax_cdf.set_title(f"{dname} Empirical CDF", fontsize=14)
    ax_cdf.set_xlabel(f"{tvar}", fontsize=12)
    ax_cdf.set_ylabel("Cumulative probability", fontsize=12)
    
    legend = ax_cdf.legend(fontsize=14, loc='lower right', shadow=True)
    for legend_handle in legend.legendHandles:
        legend_handle.set_markersize(10)
    
    
    #ax_cdf.legend(fontsize=14, loc='lower right', shadow=True)
    ax_cdf.set_xticks(xh_tick_positions)
    ax_cdf.set_yticks(tick_positions)
    ax_cdf.set_xlim(round(thresh, 2)-0.05, 1.05)
    
    if SAVE_FIGS:
        fig.savefig(FIGS_PATH.joinpath(f"extremes_dist_{res_name}.png"))

    plt.tight_layout()    
    plt.show()

def PlotResampledResults(resamp_results, extremes_idx, steps_out, thresh, tvar, dname):
    for k in resamp_results.keys():
        fig = plt.figure(figsize=(20,5))
        ax = fig.add_subplot(121)
        ax_hist = fig.add_subplot(122)

        if "GAN_CNN" in k: col_key = "GAN_CNN"
        elif "GAN_FNN" in k: col_key = "GAN_FNN"
        else: col_key = k

        res_col = adjust_brightness(strat_cols['relevance'], 0.8)
        #dataf = flatten_resampled(resamp_results[k], extremes_idx)
        dataf = resamp_results[k][:,-steps_out]
        sns.lineplot(data=dataf, ax=ax, color=strat_cols[col_key], linewidth=0.5, alpha=0.3)
        sns.scatterplot(data=dataf, ax=ax, color=res_col,edgecolor='None', marker='o', linestyle='None', s=1)
        ax.axhline(y=thresh, color='k', linestyle=':', label=f"Extreme Threshold: {round(thresh,3)}", linewidth=1.0)
        ax.set_xlabel("Time (samples)", fontsize=12)
        ax.set_ylabel(tvar, fontsize=12);
        ax.set_title(f"{dname} Resampled via {rohit(k)}", fontsize=16)
        tick_positions = np.arange(0, 1.1, 0.1)
        ax.set_yticks(tick_positions)
        ax.legend(fontsize=12, loc='upper left', shadow=True)
        
        sns.histplot(data=dataf, bins=100, color=strat_cols[col_key], ax=ax_hist)
        ax_hist.axvline(x=thresh, color='k', linestyle=':', label=f"Extreme Threshold: {round(thresh,3)}")
        ax_hist.set_title(f"{dname} Disribution Resampled via {rohit(k)}", fontsize=16)
        ax_hist.set_xlabel(tvar, fontsize=12)
        ax_hist.set_ylabel("Frequency", fontsize=12)
        ax_hist.set_xticks(tick_positions)
        ax_hist.legend(fontsize=12, loc='upper right', shadow=True)
        #ax.plot(np.arange(resamp_results[k][:,0].size), resamp_results[k][:,-steps_out],linewidth=0.5,alpha=0.5)
        #ax.scatter(np.arange(resamp_results[k][:,0].size), resamp_results[k][:,-steps_out],c='r',s=0.1,linewidth=0.5)
        #ax.axhline(y=thresh, color='k', linestyle=':', label='Thresh', linewidth=0.5)
        '''
        ax_hist.hist(resamp_results[k][:,-steps_out], bins=20)
        ax_hist.axvline(x=thresh, color='k', linestyle=':', label='Thresh')
        ax_hist.set_title(f"Training Disribution Resampled via {k}")
        ax_hist.set_xlabel(tvar)
        ax_hist.set_ylabel("Frequency")
        ax.set_title(f"{k} Resampling Results")
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel(f"{tvar}")
        ax_hist.legend([f"Extreme Threshold: {round(thresh,3)}"])
        ax.legend([f"Extreme Threshold: {round(thresh,3)}"])
        '''
        if SAVE_FIGS:
            fig.savefig(FIGS_PATH.joinpath(f"resampling_results_{k}.png"))
        plt.tight_layout()    
        plt.show()

def PlotPredVsActual1D(actual, pred, res_name, ex_thresh, tvar, dname, forecaster, combine=0):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    if combine == 'mean':
        actual = np.mean(actual, axis=1)
        pred = np.mean(pred, axis=1)
    else:
        actual = actual[:,combine]
        pred = pred[:,combine]
    if "GAN_CNN" in res_name: col_key = "GAN_CNN"
    elif "GAN_FNN" in res_name: col_key = "GAN_FNN"
    else: col_key = res_name
    act_c = adjust_brightness(strat_cols['relevance'], 0.8)
    sns.lineplot(data=actual, ax=ax, color=act_c, linewidth=1, alpha=1, label="Actual")
    sns.lineplot(data=pred, ax=ax, color=strat_cols[col_key], linewidth=1, alpha=0.5, label="Predicted")
    ax.axhline(y=ex_thresh, color='k', linestyle=':', label=f"Extreme Threshold: {round(ex_thresh,3)}", linewidth=1.0)
    
    ax.set_xlabel("Time (samples)", fontsize=12)
    ax.set_ylabel(tvar, fontsize=12);
    ax.set_title(f"{dname}, {forecaster} Actual vs Predicted, {rohit(res_name)}", fontsize=16)
    tick_positions = np.arange(0, 1.1, 0.1)
    ax.set_yticks(tick_positions)
    ax.legend(fontsize=12, loc='upper right', shadow=True)

    if SAVE_FIGS:
        fig.savefig(FIGS_FORE_PATHS[forecaster].joinpath(f"actual_vs_pred_for_{res_name}.png"))

    plt.tight_layout()
    plt.show()

def PlotPredVsActualSteps(actual, pred, res_name, ex_thresh, tvar, dname, nsteps, forecaster):
    fig = plt.figure(figsize=(7*nsteps,5))
    if "GAN_CNN" in res_name: col_key = "GAN_CNN"
    elif "GAN_FNN" in res_name: col_key = "GAN_FNN"
    else: col_key = res_name
    for i in range(nsteps):
        ax = fig.add_subplot(1, nsteps, i + 1)
        a = actual[:,i]
        p = pred[:,i]
        act_c = adjust_brightness(strat_cols['relevance'], 0.8)
        sns.lineplot(data=a, ax=ax, color=act_c, linewidth=1, alpha=1, label="Actual")
        sns.lineplot(data=p, ax=ax, color=strat_cols[col_key], linewidth=1, alpha=0.9, label="Predicted")
        ax.axhline(y=ex_thresh, color='k', linestyle=':', label=f'Extremes Thresh: {round(ex_thresh,3)}', linewidth=0.5)
        ax.set_xlabel(f'Time (samples) @ step: {i+1}', fontsize=12)
        ax.set_ylabel(tvar, fontsize=12)
        tick_positions = np.arange(0, 1.1, 0.1)
        ax.set_yticks(tick_positions)
        ax.set_title(f"{dname}, {forecaster} Actual vs Predicted, {rohit(res_name)}", fontsize=16)
        ax.legend(fontsize=12, loc='upper right', shadow=True)
    if SAVE_FIGS:
        fig.savefig(FIGS_FORE_PATHS[forecaster].joinpath(f"actual_vs_pred_for_{res_name}_nsteps.png"))
    plt.tight_layout()
    plt.show()

def PlotStepBarResults(results, metric, agg, t, dname, nsteps, forecaster):
    met_label = f"{metric}_{t}_{agg}"

    gan_cnn_k = results.loc[results['Res'].str.contains('GAN_CNN', na=False), 'Res'].iloc[0] if any(results['Res'].str.contains('GAN_CNN', na=False)) else ''
    if not gan_cnn_k: gan_cnn_k = "GAN_CNN"
    gan_fnn_k = results.loc[results['Res'].str.contains('GAN_FNN', na=False), 'Res'].iloc[0] if any(results['Res'].str.contains('GAN_FNN', na=False)) else ''
    if not gan_fnn_k: gan_fnn_k = "GAN_FNN"

    new_strat_cols = {rohit(gan_cnn_k) if key == 'GAN_CNN' else (rohit(gan_fnn_k) if key == 'GAN_FNN' else rohit(key)): value for key, value in strat_cols.items()}
    # Create a new DataFrame with one row per 'Step' and one column per 'Res'
    step_dfs = []
    for i, row in results.iterrows():
        for j, met_value in enumerate(row[met_label]):
            step_df = pd.DataFrame({'Res': [rohit(row['Res'])],
                                    'Step': [f"{j+1}"],
                                    met_label: [met_value]})
            step_dfs.append(step_df)

    # Concatenate the step DataFrames into one
    df_melted = pd.concat(step_dfs, ignore_index=True)
    # Create the clustered bar plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    sns.barplot(data=df_melted, x='Step', y=met_label, hue='Res', palette=new_strat_cols, ax=ax)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel(f"{metric}({agg})", fontsize=12)
    ax.set_title(f"{forecaster} {t} {metric}({agg}) for {dname}", fontsize=16)
    lgd = ax.legend(fontsize=10, loc=(0.96, 0.7), shadow=True)
    if SAVE_FIGS:
        fig.savefig(FIGS_FORE_PATHS[forecaster].joinpath(f"clustered_results {nsteps}stepsout {t}_{metric}({agg})_{dname}.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


def PlotTrainTestBarResults(results, metric, agg, dname, nsteps, forecaster):
    met_tr_ag = f"{metric}_Train_{agg}"
    met_te_ag = f"{metric}_Test_{agg}"
    
    gan_cnn_k = results.loc[results['Res'].str.contains('GAN_CNN', na=False), 'Res'].iloc[0] if any(results['Res'].str.contains('GAN_CNN', na=False)) else ''
    if not gan_cnn_k: gan_cnn_k = "GAN_CNN"
    gan_fnn_k = results.loc[results['Res'].str.contains('GAN_FNN', na=False), 'Res'].iloc[0] if any(results['Res'].str.contains('GAN_FNN', na=False)) else ''
    if not gan_fnn_k: gan_fnn_k = "GAN_FNN"

    new_strat_cols = {"A" if key == 'GAN_CNN' else (rohit(gan_fnn_k) if key == 'GAN_FNN' else rohit(key)): value for key, value in strat_cols.items()}
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    df_melted = pd.melt(results, id_vars=['Res'], value_vars=[met_tr_ag, met_te_ag], var_name='Type', value_name=f"{metric}_{agg}")    
    df_melted['Type'] = df_melted['Type'].replace({f"{metric}_Train_{agg}": "Train", f"{metric}_Test_{agg}": "Test"}, regex=True)
    df_melted['Res'] = df_melted['Res'].apply(rohit)
    df_melted['Res'][0] = "1D-GAN"
    df_melted['Res'][1] = "1D-GAN"
    df_melted['Res'][2] = "1D-Conv GAN"
    df_melted['Res'][3] = "SMOTE-R"
    df_melted['Res'][4] = "SMOTE-R-bin-r"
    df_melted['Res'][5] = "SMOTE-R-bin"
    df_melted['Res'][6] = "SMOTE-R-bin-tr"
    print(df_melted['Res'][0])
  
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='Type', y=f"{metric}_{agg}", hue='Res', ci=None, palette=new_strat_cols, ax=ax)
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel(f"{metric} ({agg})", fontsize=12)
    ax.set_title(f"{forecaster} {metric} ({agg}) for {dname} accross {nsteps} prediction horizons", fontsize=16)
    lgd = ax.legend(fontsize=16, loc=(0.96, 0.7), shadow=True)
    
    if SAVE_FIGS:
        fig.savefig(FIGS_FORE_PATHS[forecaster].joinpath(f"clustered_results accross {nsteps}stepsout {metric}({agg})_{dname}.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.tight_layout()
    #plt.subplots_adjust(right=15)
    plt.show()



#heatmap plotting code:
'''
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#sns.set_theme(style="white")

corr = np.corrcoef(k_X, rowvar=False)
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
'''

'''
from scipy.stats import ks_2samp
import scipy.stats as stats
#import plt_utils as pu
#importlib.reload(pu)
step_out = 0

pu.PlotKDSCompare(Y_train, resampling_results, RELEVANCE_FUNCTION, RELEVANCE_THRESHOLD, step_out, N_STEPS_IN*N_FVARS, EXTREMES_THRESHOLD, TVAR, DATA_NAME)    
pu.PlotCDFCompare(Y_train, resampling_results, RELEVANCE_FUNCTION, RELEVANCE_THRESHOLD, step_out, N_STEPS_IN*N_FVARS, EXTREMES_THRESHOLD, TVAR, DATA_NAME)    

y_orig_rel = RELEVANCE_FUNCTION.__call__(Y_train[:,step_out])
rare_mask = [val >= RELEVANCE_THRESHOLD for val in y_orig_rel]
y_orig_rare = Y_train[rare_mask]
ks_results = []
for key, res in resampling_results.items():
    y_res = res[:,(N_STEPS_IN*N_FVARS):]
    y_rel = RELEVANCE_FUNCTION.__call__(y_res[:,step_out])
    rare_mask = [val >= RELEVANCE_THRESHOLD for val in y_rel]
    y_rare = y_res[rare_mask]

    # Perform the Kolmogorov-Smirnov test
    statistic, p_value = ks_2samp(y_orig_rare[:,step_out], y_rare[:,step_out])
    print(f"KS statistic: {statistic}, p-value: {p_value}")
    ks_results.append({'Res':key, 'ksStat':statistic, 'pValue':p_value})

    pu.PlotExremesDistribution(y_orig_rare, y_rare, key, step_out, EXTREMES_THRESHOLD, TVAR, DATA_NAME)
    
if SAVE_ANY:
    ks_df = pd.DataFrame(results_list)
    ks_df.to_csv(SESSION_PATH.joinpath("ks_results.csv"))
'''

def flatten_resampled(data, extremes_idx):
    flat_original = data[:extremes_idx,0]
    flat_original = np.append(flat_original, data[extremes_idx-1][1:])
    flat_extremes = data[extremes_idx:].flatten()
    return np.append(flat_original, flat_extremes)