from collections import Counter
import numpy as np
from src.fewshot.utils import (
    compute_confidence_interval,
    load_checkpoint,
    Logger,
    extract_features,
)
from src.fewshot.methods.paddle import PADDLE
from src.fewshot.methods.soft_km import SOFT_KM
from src.fewshot.methods.baseline import Baseline
from src.fewshot.methods.tim import TIM, ALPHA_TIM
from src.fewshot.methods.simpleshot import SIMPLE_SHOT
from src.fewshot.datasets import (
    Tasks_Generator,
    SamplerSupport,
    SamplerQuery,
    CategoriesSampler,
)
import torch
import os
import matplotlib.pyplot as plt
from src.fewshot.normalize import norm_reinhard
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)
from src.fewshot.querysets_fullslide import extract_query_sets_full_slide_prediction
from tqdm import tqdm
from src.fewshot.display_predictions import display_prediction
import random
import pickle
import logging
from datetime import datetime
from src.utils.LungDataset import LungSet
import pandas as pd
from src.fewshot.datasets.task_generator import generate_support_set, generate_query_set
from src.fewshot.confusion_matrix import create_confusion_matrix
from sklearn.metrics import classification_report


class Evaluator:
    def __init__(self, device, args, log_file):
        """
        Initializes the class instance with the given device, arguments, and log file.

        Args:
            device (str): The device to use for computations.
            args (Namespace): The arguments passed to the class.
            log_file (str): The path to the log file.

        Returns:
            None
        """
        self.device = device
        self.args = args
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
    
        print(self.args)

    def run_full_evaluation(self, model):
        l = logging.getLogger("super_logger")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fileHandler = logging.FileHandler("super_test_logs.txt", mode="a")
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        l.setLevel(logging.INFO)
        l.addHandler(fileHandler)
        l.addHandler(streamHandler)

        super_logger = logging.getLogger("super_logger")
        random_number = int(
            datetime.now().timestamp()
        )  # Convert current time to seconds => generate a random number  which will be the name of the test
        super_logger.info(
            f"This is an informational message:  Test_{random_number} Started..."
        )
        self.logger.info(
            "=> Runnning full evaluation with method: {}".format(self.args.name_method)
        )

        # Load the model used as feature extractor
        load_checkpoint(model=model, model_path=self.args.features_extractor_path)

        ## load the df
        df_query = pd.read_csv(self.args.split_dir + self.args.query_split_file + '.csv').sample(frac = 1)

        df_support = pd.read_csv(self.args.split_dir + self.args.support_split_file + '.csv').sample(frac = 1)
        df_support_only_augmented = pd.read_csv(self.args.split_dir + self.args.support_split_file_only_augmented + '.csv').sample(frac = 1)


        # create the datasets
        query_dataset = LungSet(df_query, self.args.query_data_dir, file_name=True, predictions=self.args.prediction)

        support_dataset = LungSet(df_support, self.args.data_dir, file_name=True)
        support_dataset_only_augmented = LungSet(df_support_only_augmented, self.args.data_dir_augmented, file_name=True)


        ## FEATURE EXTRACTION support
        ## Create a dictionary: extracted_features_dic_support = {
        ## 'concat_features': tensor([S, 512]),                                 transformed and extracted features of the support set
        ## 'concat_labels': tensor([S]),                                        labels of the support set
        ## 'concat_slices': tensor([S]): ['15_B', '85_M', ...],                 the name of the corresponding WSI
        ## 'concat_patchs': tensor([S]): ['15_B_1_AM_0.jpg', '85_M_1_AM_0.jpg', ...]}   patch file names of the support set
        extracted_features_dic_support = extract_features(
            model=model,
            dataset=support_dataset,
            save_directory=self.args.features_dir,
            save_filename=self.args.support_split_file + '.plk'
        )

        extracted_features_dic_support_only_augmented = extract_features(
            model=model,
            dataset=support_dataset_only_augmented,
            save_directory=self.args.features_dir,
            save_filename=self.args.support_split_file_only_augmented + '.plk'
        )

        ## ''' FEATURE EXTRACTION query'''
        ## Create a dictionary: extracted_features_dic_support = {
        ## 'concat_features': tensor([Q, 512]),                                     transformed and extracted features of the support set
        ## 'concat_labels': tensor([Q]),                                            labels of the support set
        ## 'concat_slices': tensor([Q]): ['36_I', '36I', ...],                      the name of the corresponding WSI
        ## 'concat_patchs': tensor([Q]): ['36_I_x_1_y_0.jpg', '36_I_x_1_y_3.jpg', ...]}     patch file names of the support set
        extracted_features_dic_query = extract_features(
            model=model,
            dataset=query_dataset,
            save_directory=self.args.features_dir,
            save_filename=self.args.query_split_file + '.plk'
        )
  
        all_features_support = extracted_features_dic_support["concat_features"]
        all_labels_support = extracted_features_dic_support["concat_labels"].long()

        # Create the windows for the query set
        if self.args.sampling == "sliding_window":
            self.args.n_query = (self.args.window_size) ** 2

            if self.args.overlapping:
                save_path = self.args.querysets_dir + self.args.query_split_file + '_overlapping' + '.plk'
            else:
                save_path = self.args.querysets_dir + self.args.query_split_file + '.plk'

            compo_querysets = extract_query_sets_full_slide_prediction(
                patch_list=extracted_features_dic_query['concat_patchs'],
                window_size=self.args.window_size,
                save_path=save_path,
                overlapping=self.args.overlapping
            )
                    
            n_iter = len(compo_querysets["query_sets"])


        if self.args.method != "paddle":
            self.args.alphas = [0]
            self.args.gamma = 1.0


        ## If we want to experiment for many values of alpha
        for alpha in self.args.alphas:
            self.args.alpha = alpha
            n_shots = []
            accuracies = []
            f1_macro_scores = []
            f1_weighted_scores = []
            balanced_accuracy_scores = []
        
            ## if we want to do the experiment for many values of shots
            for n_shot in self.args.shots:
                global_truth = []
                global_prediction = []
                global_confidence = []
                global_patch_order = []


                # create the support set once for each task
                x_support, y_support = generate_support_set(extracted_features_dic_support, extracted_features_dic_support_only_augmented, n_shot)
    
                
                ## save or load from quey_sets_patchsize_{self.args.patch_size}_squares_{self.args.trainset_name}/query/queryset.plk
                ## reconstruct the squares as query set for the task
                ## compo_querysets is a list of list of patch names
                ## 'query_sets': [['36_I_1_AM_1', '36_I_1_AM_2', ...][]...] list of querysets, each queryset is a list of patch names}
                # elif self.args.sampling == "squares":
                #     self.args.nquery = 25
                #     querysets_folder = f"query_sets_patchsize_{self.args.patch_size}_squares_{self.args.trainset_name}"
                #     compo_querysets = extract_query_sets_full_slide_prediction(
                #         self.args.window_size,
                #         self.args,
                #         querysets_folder,
                #         self.args.trainset_name,
                #         squares=True,
                #     )
                #     n_iter = len(compo_querysets)

                # elif (
                #     self.args.number_tasks == "max" or self.args.same_query_size
                # ):  # car dans le 2e cas on veut avoir assez de tasks
                #     name_slices = np.unique(all_slices_query)
                #     n_iter = len(name_slices)
                # else:
                #     n_iter = int(self.args.number_tasks / self.args.batch_size)
                
                print("Inference started for n_shot: ", n_shot, " and alpha: ", alpha)

                ## do the inference with each query set (window)
                for i in tqdm(range(n_iter)):
                    query_set_files = compo_querysets["query_sets"][i]
                    x_query, y_query = generate_query_set(extracted_features_dic_query, query_set_files)

                    # Generate tasks
                    ## 'x_s': tensor([batch, #samples=shots*way, #features])
                    ## 'y_s': tensor([batch, #samples=shots*way, 1])
                    ## 'x_q': tensor([batch, #samples=windows_size, #features])
                    ## 'y_q': tensor([batch, #samples=windows_size, 1])}
                    tasks = {
                        "x_s": x_support,
                        "y_s": y_support,
                        "x_q": x_query,
                        "y_q": y_query,
                    }

                    # Get the method
                    ## Get the model (PADDLE) in general
                    method = self.get_method_builder(model=model)

            
                    ## Run the task with the method and returns:
                    ## logs: the logs of the method
                    ## truth: [window_size] the true labels of the task query set
                    ## pred: [window_size] the predicted labels of the task query set
                    ## conf: [window_size] the confidence of the predicted labels
                    if self.args.method == "paddle":
                        logs, truth, pred, conf = method.run_task(
                            task_dic=tasks,
                            all_features_support=all_features_support,
                            all_labels_support=all_labels_support,
                            gamma=self.args.gamma,
                            shot=n_shot,
                        )
                
                    else:
                        logs, truth, pred, conf = method.run_task(
                            task_dic=tasks, shot=n_shot
                        )

                    ## Save the predictions and confidences of each single patch and in order to reconstruct the predictions
                    global_prediction += pred
                    global_confidence += conf
                    global_patch_order += query_set_files
                    global_truth += truth

                    ## keep track of how many classes are inside a sliding window
                    #number_classes_by_query_set.append(len(set(pred)))

                    # if self.args.evaluation:
                    #     acc_mean, acc_conf = compute_confidence_interval(
                    #         logs["acc"][:, -1]
                    #     )
                    #     results_task.append(acc_mean)
                    #     global_truth += truth
                    #     if (alpha == "adaptatif_100%") and (shot >= 30):
                    #         len_query_vs_accuracy.append(
                    #             (tasks["y_q"].shape[1], acc_mean)
                    #         )

                    # if (self.args.number_tasks != "max") and (
                    #     nb_tasks_real
                    #     == nb_passages_pour_moyenner * self.args.number_tasks
                    # ):
                    #     break
                    
                    del method
                    del tasks

                # Reconstruction of the predictions using majority vote and display it
                if self.args.prediction:
                    predictions = {}

                    for i, patch in enumerate(global_patch_order):
                        if patch not in predictions:
                            predictions[patch] = []
                        predictions[patch].append(global_prediction[i])

                    for patch in predictions:
                        predictions[patch] = max(set(predictions[patch]), key=predictions[patch].count)

                    # predictions = list(predictions.values())
                    #truth = list(truth.values())
                    

                    display_prediction(
                        compo_querysets,
                        predictions,
                        save_dir=self.args.prediction_dir + self.args.query_split_file + '/',
                        save_name=f'{self.args.prefix}_{self.args.method}_{alpha}_{str(n_shot)}_shots_woverlap_{self.args.overlapping}_{self.args.covariance_used}.png',
                        title = f'{self.args.query_split_file}_{self.args.method}_{alpha}_{str(n_shot)}_shots_woverlap_{self.args.overlapping}_{self.args.covariance_used}'
                    )


                # create the confusion matrix and the classification report
                elif self.args.evaluation and (not self.args.overlapping):
                    S_all = '' if not self.args.s_use_all_train_set else '_S_all'
                    save_dir = self.args.evaluation_dir + f'/{self.args.prefix}_{self.args.method}_{alpha}_{str(n_shot)}_shots_{self.args.covariance_used}/'

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)

                    create_confusion_matrix(global_truth, global_prediction, save_dir)
                    with open(save_dir + "classification_report.txt", "w") as f:
                        f.write(classification_report(y_true=global_truth, y_pred=global_prediction, target_names=['P', 'H', 'Né', 'TL', 'Fi', 'T']))

                    accuracy = accuracy_score(global_truth, global_prediction)
                    f1_score_macro = f1_score(global_truth, global_prediction, average="macro")
                    f1_score_weighted = f1_score(global_truth, global_prediction, average="weighted")
                    balanced_accuracy = balanced_accuracy_score(global_truth, global_prediction)

                    n_shots.append(n_shot)
                    accuracies.append(accuracy)
                    f1_macro_scores.append(f1_score_macro)
                    f1_weighted_scores.append(f1_score_weighted)
                    balanced_accuracy_scores.append(balanced_accuracy)


                    print(f"Accuracy for {n_shot}-shot: {accuracy}")


                elif self.args.evaluation and self.args.overlapping:
                    S_all = '' if not self.args.s_use_all_train_set else '_S_all'
                    save_dir = self.args.evaluation_dir + f'/{self.args.prefix}_{self.args.method}_{alpha}_{str(n_shot)}_shots_woverlapping_{self.args.covariance_used}/'
                    
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)

                    predictions = {}
                    truth = {}

                    for i, patch in enumerate(global_patch_order):
                        if patch not in predictions:
                            predictions[patch] = []
                            truth[patch] = global_truth[i]
                        predictions[patch].append(global_prediction[i])

                    for patch in predictions:
                        predictions[patch] = max(set(predictions[patch]), key=predictions[patch].count)

                    predictions = list(predictions.values())
                    truth = list(truth.values())

                    create_confusion_matrix(truth, predictions, save_dir)
                    with open(save_dir + "classification_report.txt", "w") as f:
                        f.write(classification_report(y_true=truth, y_pred=predictions, target_names=['P', 'H', 'Né', 'TL', 'Fi', 'T']))

                    accuracy = accuracy_score(truth, predictions)
                    f1_score_macro = f1_score(truth, predictions, average="macro")
                    f1_score_weighted = f1_score(truth, predictions, average="weighted")
                    balanced_accuracy = balanced_accuracy_score(truth, predictions)
                    
                    n_shots.append(n_shot)
                    accuracies.append(accuracy)
                    f1_macro_scores.append(f1_score_macro)
                    f1_weighted_scores.append(f1_score_weighted)
                    balanced_accuracy_scores.append(balanced_accuracy)

                    print(f"Accuracy for {n_shot}-shot with overlapping: {accuracy}")

                else:
                    raise ValueError("To be implemented")
            
            #save the accuracies dictionary into a csv file
            if self.args.evaluation:
                save_dir = self.args.evaluation_dir

                # create a dataframe where the keys are the n_shots and the values are the accuracies, f1 macro and f1 weighted scores
                df = pd.DataFrame({
                    'n_shots': n_shots,
                    'accuracy': accuracies,
                    'f1_macro': f1_macro_scores,
                    'f1_weighted': f1_weighted_scores,
                    'balanced_accuracy': balanced_accuracy_scores
                })

                df.to_csv(self.args.evaluation_dir + f'{self.args.prefix}_{self.args.method}_woverlap_{self.args.overlapping}_{alpha}_{self.args.covariance_used}_analysis.csv', index=False)


        logging.info("----- Test has ended -----")
        return None




    def get_method_builder(self, model):
        # Initialize method classifier builder
        method_info = {
            "model": model,
            "device": self.device,
            "log_file": self.log_file,
            "args": self.args,
        }
        if self.args.name_method == "PADDLE":
            method_builder = PADDLE(**method_info)
        elif self.args.name_method == "SOFT-KM":
            method_builder = SOFT_KM(**method_info)
        elif self.args.name_method == "Baseline":
            method_builder = Baseline(**method_info)
        elif self.args.name_method == "TIM":
            method_builder = TIM(**method_info)
        elif self.args.name_method == "ALPHA_TIM":
            method_builder = ALPHA_TIM(**method_info)
        elif self.args.name_method == "SIMPLE_SHOT":
            method_builder = SIMPLE_SHOT(**method_info)
        else:
            self.logger.exception("Method must be in ['PADDLE', 'SOFT_KM', 'Baseline']")
            raise ValueError("Method must be in ['PADDLE', 'SOFT_KM', 'Baseline']")
        return method_builder
