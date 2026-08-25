



# # load a prediction and label from the pickle
# # output_file = "/Users/20759193/repos/rPPG-Toolbox/runs/physhydra/CVP_RGB_H128_W128_CHUNK128_DIFFNORM_CCC/saved_test_outputs/NECKFLIX_CVP_PHYSHYDRA_RGB_DIFFNORM_outputs.pickle"
# output_file = "test_metrics.pickle"
# with open(output_file, "rb") as f:
#     data = pickle.load(f)

# sig_type = 'abp'
# if sig_type == 'abp':
#     min_p = -50
#     max_p = 200
# elif sig_type == 'cvp':
#     min_p = -30
#     max_p = 40
# elif sig_type == 'ecg':
#     min_p = -1500
#     max_p = 1500
# else:
#     raise ValueError("Signal type not recognized for unnormalization")

# data.keys()
# predictions = data["predictions"]
# labels = data["labels"]
# fs = data["fs"]
# dicts = []
# for index in tqdm(predictions.keys(), ncols=80):
#     prediction = unnormalize_signal(_reform_data_from_dict(predictions[index]), min_p, max_p).detach().cpu().numpy()
#     label = unnormalize_signal(_reform_data_from_dict(labels[index]), min_p, max_p).detach().cpu().numpy()
#     participant, _, _, position, substr = index.split('_')
#     depth, camera, channels, trace = substr.split('-')
#     dicts.append({
#         'trace': trace,
#         'index': index,
#         'recording': index.split('-')[0],
#         'participant': participant,
#         'depth': bool(depth == 'D'),
#         'camera': camera,
#         'channels': channels,
#         'prediction': prediction,
#         'label': label
#     })

# df = pd.DataFrame(dicts)
# df.head()

# # group the dataframe by recording and get the first recording
# df.groupby(by='recording')



# stat_df



# pred_df = pd.DataFrame(prediction_stats).set_index('index')
# label_df = pd.DataFrame(label_stats).set_index('index')
# error_df = pred_df - label_df
# error_df

# mean_error = pred_df['mean'] - label_df['mean']
# mean_percentage_error = mean_error / pred_df['mean'] * 100
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.scatterplot(x=label_df['mean'], y=mean_percentage_error)
# plt.scatter(x=label_df['mean'], y=mean_error)
# plt.show()

# max_error = pred_df['max'] - label_df['max']
# max_percentage_error = max_error / label_df['max'] * 100
# plt.scatter(x=label_df['max'], y=max_percentage_error)
# plt.show()

# min_error = pred_df['min'] - label_df['min']
# min_percentage_error = min_error / label_df['min'] * 100
# plt.scatter(x=label_df['min'], y=min_percentage_error)
# plt.show()


# pred_dia_idx, pred_dia_values = find_peaks(prediction, type='min', width=width)
# pred_sys_mean = np.mean(pred_sys_values)
# pred_sys_se = np.std(pred_sys_values) / np.sqrt(len(pred_sys_values))
# pred_dia_mean = np.mean(pred_dia_values)
# pred_dia_se = np.std(pred_dia_values) / np.sqrt(len(pred_dia_values))
# label_sys_idx, label_sys_values = find_peaks(label, type='max', width=31)
# label_dia_idx, label_dia_values = find_peaks(label, type='min', width=31)
# label_sys_mean = np.mean(label_sys_values)
# label_sys_se = np.std(label_sys_values) / np.sqrt(len(label_sys_values))
# label_dia_mean = np.mean(label_dia_values)
# label_dia_se = np.std(label_dia_values) / np.sqrt(len(label_dia_values))
# print(f"Prediction Systolic Mean: {pred_sys_mean:.1f} +/- {pred_sys_se:.1f}, Label Systolic Mean: {label_sys_mean:.1f} +/- {label_sys_se:.1f}")
# print(f"Prediction Diastolic Mean: {pred_dia_mean:.1f} +/- {pred_dia_se:.1f}, Label Diastolic Mean: {label_dia_mean:.1f} +/- {label_dia_se:.1f}")


# fig, ax = plt.subplots(figsize=(12, 4))
# ax.plot(prediction.numpy(), label='Prediction')
# ax.scatter(pred_sys_idx, pred_sys_values, color='red', label='Predicted Systolic Peaks')
# ax.plot(label.numpy(), label='Label')
# ax.scatter(label_sys_idx, label_sys_values, color='green', label='Label Systolic Peaks')
# ax.set_title('PPG Signal Prediction vs. Ground Truth')
# ax.set_xlabel('Frame')
# ax.set_ylabel('PPG Signal Amplitude')
# ax.legend()
# plt.show()




# peak_values


# prediction.shape
# label.shape