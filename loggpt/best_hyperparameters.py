from bayes_opt import BayesianOptimization, UtilityFunction
from utils import utils
from model import initGPT
from loguru import logger
import pandas as pd
from utils.voting_experts import VotingExperts
from utils.utils import ve_segmentation, bayes_preprocessing, bayes_sliding_window, sliding_window, bayes_ve_segmentation, bayes_nokia_sliding_window
import numpy as np
from scipy.stats import ttest_rel
import glob
import os
from tqdm import tqdm

def bayes_for_voting_experts(args, options, segmentation_column='Node'):
    optimizer = BayesianOptimization(f=None,
                    pbounds={
                            'window': [3, 40],
                            'threshold': [1, 10],
                            },
                    verbose=2, random_state=5385)

    utility = UtilityFunction(kind='ucb', kappa=1.96, xi=0.01)
    ctr = 0

    log_structured_files = glob.glob(os.path.join(args.dest_dir, '*.log_structured'))[0:500]
    ve_models = {}
    f1s_sw = [0.95714, 0.96269, 0.96153, 0.96563, 0.96191, 0.96234, 0.96528, 0.96489, 0.96341, 0.95836]
    from tqdm import tqdm
    for log_file in tqdm(log_structured_files, desc="preprocessing log files"):
        file_name = os.path.basename(log_file)
        df = bayes_preprocessing( dataset_name=options['dataset_name'], options=options, dataframe=None, source_dir=args.dest_dir, source_file=file_name)
        grouped_by = df.groupby(segmentation_column)
        event_sequence = grouped_by['EventId'].apply(list).reset_index()
        event_dict = event_sequence.set_index(segmentation_column).to_dict()['EventId']
        ve = VotingExperts(40,10)
        ve.fit(event_dict)
        ve_models[file_name] = (ve, grouped_by, event_dict)
    

    for _ in tqdm(range(int(args.iterations)), desc="Bayesian Optimization"):
        next_point = optimizer.suggest(utility)
        ctr += 1
        window = int(next_point['window'])
        threshold = int(next_point['threshold'])
        options['window_size'] = window
        options['threshold'] = threshold

        dataframes = []
        for file_name in ve_models.keys():
            ve, grouped_by, event_dict = ve_models[file_name]
            # ve = VotingExperts(window, threshold)
            # ve.fit(event_dict)
            new_df = bayes_ve_segmentation(grouped_by,event_dict, options, segmentation_method=ve)
            dataframes.append(new_df)

        new_df = pd.concat(dataframes, ignore_index=True)
        # new_df.to_csv(args.dest_dir/'BGL.W{}.T{}.csv'.format(options['window_size'],
        #                                                       options['threshold']))
        f1s = []
        precs = []
        recs = []

        for _ in range(10):
            train_df, test_df = utils.train_test_split(options['dataset_name'], 
                                options['train_samples'], 
                                options['seed'], 
                                options=options, 
                                source_dir=args.dest_dir,
                                dataframe=new_df)
            initGPT_model = initGPT.InitGPT(options, train_df, test_df)
            f1s.append(initGPT_model.f1)
            precs.append(initGPT_model.prec)
            recs.append(initGPT_model.rec)
            logger.info(f"f1 {initGPT_model.f1}")
            logger.info(f"prec {initGPT_model.prec}")
            logger.info(f"rec {initGPT_model.rec}")
        f1_mean = np.mean(f1s)
        f1_std = np.std(f1s)
        prec_mean = np.mean(precs)
        prec_std = np.std(precs)
        rec_mean = np.mean(recs)
        rec_std = np.std(recs)
        #t_stat_ve, p_value_ve = ttest_rel(f1s_sw, f1s)

        #mlflow.log_metric("p_value_ve", p_value_ve)
        logger.info(f"for window: {window} and threshold {threshold} f1 mean is {f1_mean}")
        logger.info(f"for window: {window} and threshold {threshold} f1 std is {f1_std}")
        #logger.info(f"for window: {window} and threshold {threshold} f1 p_value is {p_value_ve}")
        logger.info(f"for window: {window} and threshold {threshold} f1 values are {f1s}")
        
        logger.info(f"for window: {window} and threshold {threshold} precision mean is {prec_mean}")
        logger.info(f"for window: {window} and threshold {threshold} precision std is {prec_std}")
        logger.info(f"for window: {window} and threshold {threshold} precision values are {precs}")

        logger.info(f"for window: {window} and threshold {threshold} recall mean is {rec_mean}")
        logger.info(f"for window: {window} and threshold {threshold} recall std is {rec_std}")
        logger.info(f"for window: {window} and threshold {threshold} recall values are {recs}")

        optimizer.register(params=next_point, target=f1_mean)
    logger.info('Best result: {}; f(x)={}.'.format(optimizer.max['params'], optimizer.max['target']))
    max_params=optimizer.max['params']
    window = max_params['window']
    threshold = max_params['threshold']
    f1 = optimizer.max['target']
    return f1, window, threshold

def bayes_for_sliding_window(args, options, segmentation_column='Node'):
    optimizer = BayesianOptimization(f=None,
                    pbounds={
                            'window': [30, 512],
                            },
                    verbose=2, random_state=5385)

    utility = UtilityFunction(kind='ucb', kappa=1.96, xi=0.01)
    ctr = 0
    df = bayes_preprocessing( dataset_name=options['dataset_name'], options=options, dataframe=None, source_dir=args.dest_dir)
    # start_time = df.timestamp.min()
    # end_time = df.timestamp.max()
    grouped_by = df.groupby(segmentation_column)
    event_sequence = grouped_by['EventId'].apply(list).reset_index()
    event_dict = event_sequence.set_index(segmentation_column).to_dict()['EventId']
    # f1s_sw = [0.95714, 0.96269, 0.96153, 0.96563, 0.96191, 0.96234, 0.96528, 0.96489, 0.96341, 0.95836]
    from tqdm import tqdm
    for _ in tqdm(range(int(args.iterations)), desc="Bayesian Optimization"):
        next_point = optimizer.suggest(utility)
        ctr += 1
        window = int(next_point['window'])
        options['window_size'] = window
        options['step_size'] = window // 2

        new_df = bayes_nokia_sliding_window(grouped_by, options)

        if new_df is None:
            logger.info("new dataframe is None")
        else:
            logger.info("new dataframe is NOT None")
        # new_df.to_csv(args.dest_dir/'BGL.W{}.T{}.csv'.format(options['window_size'],
        # 
        #                                                      options['threshold']))
        f1s = []
        for _ in range(10):
            train_df, test_df = utils.train_test_split(options['dataset_name'], 
                                options['train_samples'], 
                                options['seed'], 
                                options=options, 
                                source_dir=args.dest_dir,
                                dataframe=new_df)
            initGPT_model = initGPT.InitGPT(options, train_df, test_df)
            f1s.append(initGPT_model.f1)
            logger.info(f"f1 {initGPT_model.f1}")
            f1_mean = np.mean(f1s)
            f1_std = np.std(f1s)
            logger.info(f"for window: {window} f1 mean is {f1_mean}, f1 std is {f1_std} ")
            logger.info(f"f1 values {f1s}")

        #t_stat_ve, p_value_ve = ttest_rel(f1s_sw, f1s_ve)

        optimizer.register(params=next_point, target=f1_mean)
    logger.info('Best result: {}; f(x)={}.'.format(optimizer.max['params'], optimizer.max['target']))
    max_params=optimizer.max['params']
    window = max_params['window']
    f1 = optimizer.max['target']
    logger.info(f"best f1 mean({f1}) was achieved for window {window}")
    return f1, window
