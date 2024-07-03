#© 2023 Nokia
#Licensed under the Creative Commons Attribution Non Commercial 4.0 International license
#SPDX-License-Identifier: CC-BY-NC-4.0
#

import random
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import regex as re
from tqdm import tqdm
from ast import literal_eval
from utils.voting_experts import VotingExperts
from datetime import datetime, timedelta


def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def hdfs_blk_process(df, blk_label_dict):
    data_dict = defaultdict(list)
    for idx, row in tqdm(df.iterrows()):
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if blk_Id not in data_dict:
                data_dict[blk_Id] = [row['EventId']]
            else:
                data_dict[blk_Id].append(row["EventId"])

    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

    data_df["Label"] = data_df["BlockId"].apply(
        lambda x: blk_label_dict.get(x))  # add label to the sequence of each blockid

    return data_df

def bayes_ve_segmentation(grouped_by_node, event_dict, options, segmentation_method=None):
    
    segmentation = segmentation_method.transform(event_dict, options['window_size'],options['threshold'])
    
    new_data = []
    for group in grouped_by_node:
        start_idx = 0
        for segment in segmentation[group[0]]:
            
            df_window = group[1].iloc[start_idx:start_idx+len(segment)]
            if len(df_window) > 1:
                if len(df_window) > 512:

                    start = 0
                    end = len(segment)
                    

                    while (end - start) > 512:

                        df_window_inner = df_window.iloc[start:start+512]

                        new_data.append([
                            df_window_inner['Label'].values.tolist(),
                            df_window_inner['Label'].max(),
                            df_window_inner['EventId'].values.tolist()
                        ])
                        start += 512 // 2
                    if start < end:
                        df_window = df_window.iloc[start:end]
                        new_data.append([
                            df_window['Label'].values.tolist(),
                            df_window['Label'].max(),
                            df_window['EventId'].values.tolist()
                        ])

                    start_idx += len(segment)
                else:
                    new_data.append([
                        df_window['Label'].values.tolist(),
                        df_window['Label'].max(),
                        df_window['EventId'].values.tolist()
                    ])
                    start_idx += len(segment) 

    #print('there are %d instances (voting experts segments) in this dataset\n' % len(new_data))
    return pd.DataFrame(new_data, columns=['Label_org', 'Label', 'EventSequence'])

def ve_segmentation(df, options, segmentation_method=None):
    if options['dataset_name'] == 'BGL':
        df['datatime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
    if options['dataset_name'] == 'Thunderbird':
        df['datatime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S')
    if options['dataset_name'] == 'OpenStack':
        df['datatime'] = pd.to_datetime(df['Time'] + ' ' + df['Pid'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        df['datatime'] = df['datatime'].fillna(method='ffill')
    df['timestamp'] = df['datatime'].values.astype(np.int64) // 10 ** 9
    df = df.sort_values('timestamp')

    df.set_index('timestamp', drop=False, inplace=True)
    start_time = df.timestamp.min()
    end_time = df.timestamp.max()
    grouped_by_node = df.groupby('Node')
    event_sequence = grouped_by_node['EventId'].apply(list).reset_index()
    event_dict = event_sequence.set_index('Node').to_dict()['EventId']
    if segmentation_method is None:
        print()
        ve = VotingExperts(options['window_size'],options['threshold'])
        segmentation = ve.fit_transform(event_dict)
    else:
        segmentation = segmentation_method.transform(event_dict, options['window_size'],options['threshold'])
    import pickle
    # with open("segmentation.pkl", "wb") as out:
    #     print("segmentation saved to segmentation.pkl")
    #     pickle.dump(segmentation, out)
    new_data = []
    for group in grouped_by_node:
        start_idx = 0
        for segment in segmentation[group[0]]:
            
            df_window = group[1].iloc[start_idx:start_idx+len(segment)]
            if len(df_window) > 1:
                if len(df_window) > 512:

                    start = 0
                    end = len(segment)
                    

                    while (end - start) > 512:

                        df_window_inner = df_window.iloc[start:start+512]

                        new_data.append([
                            df_window_inner['Label'].values.tolist(),
                            df_window_inner['Label'].max(),
                            df_window_inner['EventId'].values.tolist()
                        ])
                        start += 512 // 2
                    if start < end:
                        df_window = df_window.iloc[start:end]
                        new_data.append([
                            df_window['Label'].values.tolist(),
                            df_window['Label'].max(),
                            df_window['EventId'].values.tolist()
                        ])

                    start_idx += len(segment)
                else:
                    new_data.append([
                        df_window['Label'].values.tolist(),
                        df_window['Label'].max(),
                        df_window['EventId'].values.tolist()
                    ])
                    start_idx += len(segment) 

    print('there are %d instances (voting experts segments) in this dataset\n' % len(new_data))
    return pd.DataFrame(new_data, columns=['Label_org', 'Label', 'EventSequence'])

def bayes_nokia_sliding_window(df, options):

    new_data = []
    for group in df:
        start_idx = 0   
        df_window = group[1]
        if len(df_window) > 1:
            if len(df_window) > options['window_size']:
                start = 0
                end = len(df_window)
            
                while (end - start) > options['window_size']:
                    df_window_inner = df_window.iloc[start:start+512]

                    new_data.append([
                        df_window_inner['Label'].values.tolist(),
                        df_window_inner['Label'].max(),
                        df_window_inner['EventId'].values.tolist()
                    ])
                    start += options['step_size']
                if start < end:
                    df_window = df_window.iloc[start:end]
                    new_data.append([
                        df_window['Label'].values.tolist(),
                        df_window['Label'].max(),
                        df_window['EventId'].values.tolist()
                    ])

                start_idx += len(df_window)
            else:
                new_data.append([
                    df_window['Label'].values.tolist(),
                    df_window['Label'].max(),
                    df_window['EventId'].values.tolist()
                ])
                start_idx += len(df_window) 

    print('there are %d instances (sliding windows) in this dataset\n' % len(new_data))
    return pd.DataFrame(new_data, columns=['Label_org', 'Label', 'EventSequence'])

def bayes_sliding_window(df, options):

    new_data = []
    for group in df:
        start_time = group[1].timestamp.min()
        end_time = group[1].timestamp.max()
        while start_time < end_time:
            df_window = group[1].loc[start_time:start_time+options["window_size"]]
            if len(df_window) > 1:  # Only consider windows with more than one value
                if len(df_window) > options['max_lens']:
                    start_time_inner = df_window.timestamp.min()
                    end_time_inner = df_window.timestamp.max()
                    while (end_time_inner - start_time_inner) > options['max_lens']:
                        df_window_inner = df_window.loc[start_time_inner:start_time_inner+options['max_lens']]
                        new_data.append([
                            df_window_inner['Label'].values.tolist(),
                            df_window_inner['Label'].max(),
                            df_window_inner['EventId'].values.tolist()
                        ])
                        start_time_inner += options['max_lens'] // 2
                    if start_time_inner < end_time_inner:
                        df_window_inner = df_window.loc[start_time_inner:end_time_inner]
                        new_data.append([
                            df_window_inner['Label'].values.tolist(),
                            df_window_inner['Label'].max(),
                            df_window_inner['EventId'].values.tolist()
                        ])
                else:
                    new_data.append([
                        df_window['Label'].values.tolist(),
                        df_window['Label'].max(),
                        df_window['EventId'].values.tolist()
                    ])
            start_time += options['step_size']

    print('there are %d instances (sliding windows) in this dataset\n' % len(new_data))
    return pd.DataFrame(new_data, columns=['Label_org', 'Label', 'EventSequence'])

def sliding_window(df, options, segmentation_column):
    # df.reset_index(drop=True, inplace=True)
    # if options['dataset_name'] == 'BGL':
    #     df['datatime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
    # if options['dataset_name'] == 'Thunderbird':
    #     df['datatime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S')
    # if options['dataset_name'] == 'OpenStack':
    #     df['datatime'] = pd.to_datetime(df['Time'] + ' ' + df['Pid'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    #     df['datatime'] = df['datatime'].fillna(method='ffill')
    # df['timestamp'] = df['datatime'].values.astype(np.int64) // 10 ** 6
    # assert df['timestamp'].dtype == np.int64, "Timestamp column is not int64"
    # df = df.sort_values('timestamp')

    # df.set_index('timestamp', drop=False, inplace=True)
    start_time = df.timestamp.min()
    end_time = df.timestamp.max()

    new_data = []
    too_small = 0
    for group in tqdm(df.groupby(segmentation_column), desc='sliding window on threads'):
        start_time = group[1].timestamp.min()
        end_time = group[1].timestamp.max()

        while start_time <= end_time:

            df_window = group[1].loc[start_time:start_time+options["window_size"]]
            if len(df_window) > 1:  # Only consider windows with more than one value

                if len(df_window) > options['max_lens']:
                    start = 0
                    end = len(df_window)
                
                    while (end - start) > options['max_lens']:
                        df_window_inner = df_window.iloc[start:start+512]

                        new_data.append([
                            df_window_inner['Label'].values.tolist(),
                            df_window_inner['Label'].max(),
                            df_window_inner['EventId'].values.tolist()
                        ])
                        start += 256
                    if start < end:
                        df_window = df_window.iloc[start:end]
                        new_data.append([
                            df_window['Label'].values.tolist(),
                            df_window['Label'].max(),
                            df_window['EventId'].values.tolist()
                        ])
                else:
                    new_data.append([
                        df_window['Label'].values.tolist(),
                        df_window['Label'].max(),
                        df_window['EventId'].values.tolist()
                    ])
            else:
                too_small += 1

            next_times = group[1].index[group[1].index > start_time + options["window_size"]//2]
            if not next_times.empty:
                start_time = next_times[0]
            else:
                break
    print('there are %d too small threads\n' % too_small)
    print('there are %d instances (sliding windows) in this dataset\n' % len(new_data))
    return pd.DataFrame(new_data, columns=['Label_org', 'Label', 'EventSequence'])

def is_valid_format(ts):
    ts = ts.strip('<>')
    try:
        pd.to_datetime(ts, format='%Y-%m-%dT%H:%M:%S.%fZ')
        return True
    except ValueError:
        return False

def decrement_timestamp(ts):
    ts = ts.strip('<>')
    dt = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.%fZ')
    dt -= timedelta(milliseconds=1)
    return f"<{dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')}>"

def increment_timestamp(ts):
    ts = ts.strip('<>')
    dt = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.%fZ')
    dt += timedelta(milliseconds=1)
    return f"<{dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')}"

def fix_timestamp(df):
    # Initialize the FixedTime column with original values
    df['FixedTime'] = df['Time'].apply(lambda x: x.strip('<>') if is_valid_format(x.strip('<>')) else None)

    # Iterate through the DataFrame to fix invalid timestamps
    for i in range(len(df)):
        if df.at[i, 'FixedTime'] is None:
            if i > 0 and df.at[i - 1, 'FixedTime'] is not None:
                df.at[i, 'FixedTime'] = increment_timestamp(df.at[i - 1, 'FixedTime'])
            elif i < len(df) - 1 and is_valid_format(df.at[i + 1, 'Time'].strip('<>')):
                df.at[i, 'FixedTime'] = decrement_timestamp(df.at[i + 1, 'Time'])
            else:
                # If there's no previous valid timestamp, we can handle this case separately
                df.at[i, 'FixedTime'] = None
    return df[df['FixedTime'].notna()]

def bayes_preprocessing( dataset_name='HDFS', options=None, dataframe=None, source_dir=".", source_file=None):
    if dataset_name == 'HDFS':
        print("Preprocessing HDFS dataset")
        df = pd.read_csv(source_dir/'HDFS.log_structured.csv', engine='c', na_filter=False, memory_map=True)
        blk_df = pd.read_csv(source_dir/'anomaly_label.csv', engine='c', na_filter=False, memory_map=True)
        blk_label_dict = {}
        for _, row in tqdm(blk_df.iterrows()):
            blk_label_dict[row['BlockId']] = 1 if row['Label'] == 'Anomaly' else 0

        df = hdfs_blk_process(df, blk_label_dict)
        print('There are %d instances in this dataset\n' % len(df))
        
    elif dataset_name == 'BGL':
        print("Preprocessing BGL dataset")
        if dataframe is None:
            df = pd.read_csv(source_dir/'BGL.log_structured.csv', engine='c', na_filter=False, memory_map=True)
            df['Label'] = df['Label'].ne('-').astype(int)
        else:
            df = dataframe
        print('There are %d instances in this dataset\n' % len(df))
        
    elif dataset_name == 'NOKIA':
        # print("Preprocessing Nokia dataset")
        if source_file is None:
            source_file = 'NOKIA.log_structured.csv'
            
        if dataframe is None:
            df = pd.read_csv(source_dir/source_file, engine='c', na_filter=False, sep=';', memory_map=True, header=None)
            # TODO: check if file contains timestamps and change this ifs
            df_len = len(df.columns)
            if source_file == 'NOKIA.log_structured.csv':
                if df_len == 6: 
                    df.columns = ['LineId', 'Label', 'Time', 'PRID_ThreadID', 'Level', 'EventId']
                elif df_len == 5:
                    df.columns = ['LineId', 'Label', 'PRID_ThreadID', 'Level', 'EventId']
            else:
                if df_len == 5: 
                    df.columns = ['Label', 'Time', 'PRID_ThreadID', 'Level', 'EventId']
                elif df_len == 4:
                    df.columns = ['Label', 'PRID_ThreadID', 'Level', 'EventId']
            df['Label'] = df['Label'].ne('-').astype(int)
        else:
            df = dataframe
        # print('There are %d instances in this dataset\n' % len(df))
        
    elif dataset_name == 'Thunderbird':
        print("Preprocessing Thunderbird dataset")
        df = pd.read_csv(source_dir/'Thunderbird.log_structured.csv', engine='c', na_filter=False, memory_map=True)
        df['Label'] = df['Label'].ne('-').astype(int)
        print('There are %d instances in this dataset\n' % len(df))
        

    elif dataset_name == 'OpenStack':
        print("Preprocessing OpenStack dataset")
        df = pd.read_csv(source_dir/'OpenStack.log_structured.csv', engine='c', na_filter=False, memory_map=True)
        with open(source_dir/'OpenStack_anomaly_labels.txt', 'r') as f:
            abnormal_label = f.readlines()
        lst_abnormal_label = []
        for i in abnormal_label[2:]:
            lst_abnormal_label.append(i.strip())
        df['Label'] = 0
        for i in range(len(lst_abnormal_label)):
            for j in range(len(df)):
                if lst_abnormal_label[i] in df['Content'][j]:
                    df['Label'][j] = 1
        print(df['Label'].value_counts())
        print('There are %d instances in this dataset\n' % len(df))

    df = df[df['EventId'] != '']
    df['EventId'] = df['EventId'].astype(str)
    if options['dataset_name'] == 'BGL':
        df['datatime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
    if options['dataset_name']=='NOKIA':
        valid_rows = df[df['Time'].str.strip() != '']
        # Apply the fix function and only keep the corrected timestamps
        df = fix_timestamp(valid_rows)
        df['Time'] = df['Time'].str.strip('<>')
        df['np_datetime'] = df['Time'].apply(np.datetime64)
        np_datetime = df['np_datetime'].to_numpy()
        epoch = np.datetime64(df['np_datetime'].iloc[0])
        timestamp = []
        for i  in range(len(np_datetime)):
            timestamp.append(np_datetime[i] - epoch)
        df['timestamp'] = np.array(timestamp).astype(np.int64)
    if options['dataset_name'] == 'Thunderbird':
        df['datatime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S')
    if options['dataset_name'] == 'OpenStack':
        df['datatime'] = pd.to_datetime(df['Time'] + ' ' + df['Pid'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        df['datatime'] = df['datatime'].fillna(method='ffill')
    #df['timestamp'] = df['datatime'].values.astype(np.int64) // 10 ** 9
    df = df.sort_values('timestamp')

    df.set_index('timestamp', drop=False, inplace=True)

    return df

def preprocessing(preprocessing=True, dataset_name='HDFS', options=None, dataframe=None, source_dir="."):
    if preprocessing:
        if dataset_name == 'HDFS':
            print("Preprocessing HDFS dataset")
            df = pd.read_csv(source_dir/'HDFS.log_structured.csv', engine='c', na_filter=False, memory_map=True)
            blk_df = pd.read_csv(source_dir/'anomaly_label.csv', engine='c', na_filter=False, memory_map=True)
            blk_label_dict = {}
            for _, row in tqdm(blk_df.iterrows()):
                blk_label_dict[row['BlockId']] = 1 if row['Label'] == 'Anomaly' else 0

            hdfs_df = hdfs_blk_process(df, blk_label_dict)
            hdfs_df.to_csv(source_dir/'HDFS.BLK.csv')
            del df
            del blk_label_dict
            del blk_df
        elif dataset_name == 'BGL':
            print("Preprocessing BGL dataset")
            if dataframe is None:
                df = pd.read_csv(source_dir/'BGL.log_structured.csv', engine='c', na_filter=False, memory_map=True)
                df['Label'] = df['Label'].ne('-').astype(int)
            else:
                df = dataframe
            print('There are %d instances in this dataset\n' % len(df))
            
            if options['sliding_window']:
                new_df = sliding_window(df, options)
                new_df.to_csv(source_dir/'BGL.W{}.S{}.csv'.format(options['window_size'],
                                                              options['step_size']))
            if options['segmentation']:
                logger.info("segmentation with voting exeprts")
                new_df = ve_segmentation(df, options)
                new_df.to_csv(source_dir/'BGL.W{}.T{}.csv'.format(options['window_size'],
                                                              options['threshold']))
            del new_df

        elif dataset_name == 'Thunderbird':
            print("Preprocessing Thunderbird dataset")
            df = pd.read_csv(source_dir/'Thunderbird.log_structured.csv', engine='c', na_filter=False, memory_map=True)
            df['Label'] = df['Label'].ne('-').astype(int)
            print('There are %d instances in this dataset\n' % len(df))
            new_df = sliding_window(df, options)
            new_df.to_csv(source_dir/'Thunderbird.W{}.S{}.csv'.format(options['window_size'],
                                                                      options['step_size']))
            del new_df

        elif dataset_name == 'OpenStack':
            print("Preprocessing OpenStack dataset")
            df = pd.read_csv(source_dir/'OpenStack.log_structured.csv', engine='c', na_filter=False, memory_map=True)
            with open(source_dir/'OpenStack_anomaly_labels.txt', 'r') as f:
                abnormal_label = f.readlines()
            lst_abnormal_label = []
            for i in abnormal_label[2:]:
                lst_abnormal_label.append(i.strip())
            df['Label'] = 0
            for i in range(len(lst_abnormal_label)):
                for j in range(len(df)):
                    if lst_abnormal_label[i] in df['Content'][j]:
                        df['Label'][j] = 1
            print(df['Label'].value_counts())


            print('There are %d instances in this dataset\n' % len(df))
            new_df = sliding_window(df, options)
            new_df.to_csv(source_dir/'OpenStack.W{}.S{}.csv'.format(options['window_size'],
                                                                      options['step_size']))
            del new_df

def train_test_split(dataset_name='HDFS', train_samples=5000, seed=42, options=None, source_dir='.', dataframe=None):
    if dataset_name == 'HDFS':
        hdfs_df = pd.read_csv(source_dir / 'HDFS.BLK.csv', index_col=0, dtype={'BlickId': str})
        df['Label'] = df['Label'].astype(int)
        hdfs_df.EventSequence = hdfs_df.EventSequence.apply(literal_eval)
        normal_df = hdfs_df[hdfs_df['Label'] == 0]
        normal_df = normal_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        anomaly_df = hdfs_df[hdfs_df['Label'] == 1]
        train_df = normal_df[:train_samples]
        test_df = pd.concat([normal_df[train_samples:train_samples+2500], anomaly_df[:2500]], ignore_index=True)
        train_df.to_csv(source_dir / 'HDFS.BLK.train.csv')
        test_df.to_csv(source_dir / 'HDFS.BLK.test.csv')
        print(f'datasets contains: {len(hdfs_df)} blocks, {len(normal_df)} normal blocks, '
              f'{len(anomaly_df)} anomaly blocks')
        print(f'Trianing dataset contains: {len(train_df)} blocks')
        print(f'Testing dataset contains: {len(test_df)} blocks, '
              f'{len(test_df.loc[test_df["Label"] == 0])} normal blocks ,{len(anomaly_df)} anomaly blocks')
        return train_df, test_df
    elif dataset_name == 'BGL' or dataset_name == 'NOKIA':
        if dataframe is None:
            if options['sliding_window']:
                df = pd.read_csv(source_dir / 'BGL.W{}.S{}.csv'.format(options['window_size'], options['step_size']), index_col=0)
            elif options['segmentation']:
                df = pd.read_csv(source_dir / 'BGL.W{}.T{}.csv'.format(options['window_size'], options['threshold']), index_col=0)
            df['Label'] = df['Label'].astype(int)
            df.EventSequence = df.EventSequence.apply(literal_eval)
        else:
            df = dataframe
        normal_df = df[df['Label'] == 0]
        normal_df = normal_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        anomaly_df = df[df['Label'] == 1]
        train_df = normal_df[:train_samples]
        test_df = pd.concat([normal_df[train_samples:], anomaly_df], ignore_index=True)
        train_df.to_csv(source_dir / 'BGL.W{}.S{}.train.csv'.format(options['window_size'], options['step_size']))
        test_df.to_csv(source_dir / 'BGL.W{}.S{}.test.csv'.format(options['window_size'], options['step_size']))
        print(f'datasets contains: {len(df)} windows, {len(normal_df)} normal windows, '
                f'{len(anomaly_df)} anomaly windows')
        print(f'Trianing dataset contains: {len(train_df)} windows')
        print(f'Testing dataset contains: {len(test_df)} windows, '
                f'{len(test_df.loc[test_df["Label"] == 0])} normal windows ,{len(anomaly_df)} anomaly windows')
        return train_df, test_df
    elif dataset_name == 'Thunderbird':
        if dataframe is None:
            df = pd.read_csv(source_dir / 'Thunderbird.W{}.S{}.csv'.format(options['window_size'], options['step_size']), index_col=0)
            df['Label'] = df['Label'].astype(int)
            df.EventSequence = df.EventSequence.apply(literal_eval)
        else:
            df = dataframe
        normal_df = df[df['Label'] == 0]
        normal_df = normal_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        anomaly_df = df[df['Label'] == 1]
        train_df = normal_df[:train_samples]
        test_df = pd.concat([normal_df[train_samples:], anomaly_df], ignore_index=True)
        train_df.to_csv(source_dir / 'Thunderbird.W{}.S{}.train.csv'.format(options['window_size'], options['step_size']))
        test_df.to_csv(source_dir / 'Thunderbird.W{}.S{}.test.csv'.format(options['window_size'], options['step_size']))
        print(f'datasets contains: {len(df)} windows, {len(normal_df)} normal windows, '
                f'{len(anomaly_df)} anomaly windows')
        print(f'Trianing dataset contains: {len(train_df)} windows')
        print(f'Testing dataset contains: {len(test_df)} windows, '
                f'{len(test_df.loc[test_df["Label"] == 0])} normal windows ,{len(anomaly_df)} anomaly windows')
        return train_df, test_df
    elif dataset_name == 'OpenStack':
        if dataframe is None:
            df = pd.read_csv(source_dir / '/OpenStack.W{}.S{}.csv'.format(options['window_size'], options['step_size']), index_col=0)
            df['Label'] = df['Label'].astype(int)
            df.EventSequence = df.EventSequence.apply(literal_eval)
        else:
            df = dataframe
        normal_df = df[df['Label'] == 0]
        normal_df = normal_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        anomaly_df = df[df['Label'] == 1]
        train_df = normal_df[:train_samples]
        test_df = pd.concat([normal_df[train_samples:], anomaly_df], ignore_index=True)
        # train_df.to_csv(source_dir / '/OpenStack.W{}.S{}.train.csv'.format(options['window_size'], options['step_size']))
        # test_df.to_csv(source_dir / '/OpenStack.W{}.S{}.test.csv'.format(options['window_size'], options['step_size']))
        print(f'datasets contains: {len(df)} windows, {len(normal_df)} normal windows, '
                f'{len(anomaly_df)} anomaly windows')
        print(f'Trianing dataset contains: {len(train_df)} windows')
        print(f'Testing dataset contains: {len(test_df)} windows, '
                f'{len(test_df.loc[test_df["Label"] == 0])} normal windows ,{len(anomaly_df)} anomaly windows')
        return train_df, test_df




def get_training_dictionary(df):
    '''Get training dictionary

    Arg:
        df: dataframe of preprocessed sliding windows

    Return:
        dictionary of training datasets
    '''
    dic = {}
    count = 0
    for i in range(len(df)):
        lst = list(df['EventSequence'].iloc[i])
        for j in lst:
            if j in dic:
                pass
            else:
                dic[j] = str(count)
                count += 1
    return dic
