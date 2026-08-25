
# set matplotlib backend to allow plots to be shown



# from evaluation.post_process import _detrend, _next_power_of_2, _calculate_SNR
# from evaluation.BlandAltmanPy import BlandAltman
# from scipy.signal import butter
# from sklearn.metrics import f1_score, precision_recall_fscore_support
# from evaluation.metrics import calculate_metrics, _reform_data_from_dict
# from evaluation.post_process import _detrend, _next_power_of_2, _calculate_SNR
# from tqdm import tqdm
# from evaluation.BlandAltmanPy import BlandAltman









trace = config.TEST.DATA.PREPROCESS.NECKFLIX.TRACES[0]


df

sns.scatterplot(data=df, x='pred_max', y='label_max', hue='label_hr')

grouped_df = df.groupby('video').agg({
    'label': lambda x: np.concatenate(x.values),
    'prediction': lambda x: np.concatenate(x.values),
    'label_mean': lambda x: np.concatenate([x.values]),
    'pred_mean': lambda x: np.concatenate([x.values]),
    'label_max': lambda x: np.concatenate([x.values]),
    'pred_max': lambda x: np.concatenate([x.values]),
    'label_min': lambda x: np.concatenate([x.values]),
    'pred_min': lambda x: np.concatenate([x.values]),
    'label_hr': lambda x: np.concatenate([x.values]),
    'pred_hr': lambda x: np.concatenate(x.values),
}).reset_index()






## aggregate by video for the bland altman plot
df.head()
df['prediction'][0]
len(df['pred_max'][0])
len(df['label_max'][0])
for i in range(len(df)):
    _ = bp_metrics.get_rmse(df['prediction'][i], df['label'][i])
    _ = bp_metrics.get_rmse(df['pred_max'][i], df['label_max'][i])
    