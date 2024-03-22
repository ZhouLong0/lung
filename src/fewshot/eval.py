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
from src.fewshot.display_my_predictions import display_prediction
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

        # # normalize
        # basic_normalisation = False
        # grayscale = False
        # if self.args.normalisation_couleur == "reinhard":
        #     ## Create and insert new reinard normalised images in the dataset (just change arg.data_path if already done)
        #     ## new normalized imagaes in the folder */data/liver_normalized_reinhard
        #     ## also change the arg.data_path to the new normalized one
        #     norm_reinhard(args=self.args)

        # ## for other normalisation methods just do it during the feature extraction
        # elif self.args.normalisation_couleur == "basic":
        #     basic_normalisation = True
        # elif self.args.normalisation_couleur == "grayscale":
        #     grayscale = True
        # elif self.args.normalisation_couleur == "no":
        #     pass
        # else:
        #     print("Unknown color normalisation:", self.args.normalisation_couleur)
        #     raise Exception("Unknown color normalisation")

        ## we want to create sliding_windows in the entire query image for the prediction
        # if self.args.sampling == "sliding_window":  # predict on WSI
        #     self.args.prediction = True
        #     self.args.evaluation = False

        # ## we only want to evaluate on the known patches of the query image
        # elif self.args.sampling in ["same_slice", "squares"]:  # eval on known squares
        #     self.args.prediction = False
        #     self.args.evaluation = True

        ## load the df
        df_query = pd.read_csv(self.args.split_dir + self.args.query_split_file + '.csv').sample(frac = 1)
        df_support = pd.read_csv(self.args.split_dir + self.args.support_split_file + '.csv').sample(frac = 1)
        df_support_only_augmented = pd.read_csv(self.args.split_dir + self.args.support_split_file_only_augmented + '.csv').sample(frac = 1)


        query_dataset = LungSet(df_query, self.args.data_dir, file_name=True)
        support_dataset = LungSet(df_support, self.args.data_dir, file_name=True)
        support_dataset_only_augmented = LungSet(df_support_only_augmented, self.args.data_dir_augmented, file_name=True)

        ## ''' FEATURE EXTRACTION support'''
        ## Create a dictionary: extracted_features_dic_support = {
        ## 'concat_features': tensor([S, 512]),                                 transformed and extracted features of the support set
        ## 'concat_labels': tensor([S]),                                        labels of the support set
        ## 'concat_slices': tensor([S]): ['15_B', '85_M', ...],                 the name of the corresponding WSI
        ## 'concat_patchs': tensor([S]): ['15_B_1_AM_0', '85_M_1_AM_0', ...]}   patch names of the support set
        extracted_features_dic_support = extract_features(
            model=model,
            dataset=support_dataset,
            save_directory=self.args.features_dir,
            save_filename=self.args.support_split_file + '.plk'
        )

        extract_features_dic_support_only_augmented = extract_features(
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
        ## 'concat_patchs': tensor([Q]): ['36_I_x_1_y_0', '36_I_x_1_y_3', ...]}     patch names of the support set
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

        #if self.args.evaluation:
            # name_file = f'results/test/{self.args.dataset}/{self.args.arch}/{self.args.name_method}.txt'
            # name_file = f"results/test/{self.args.dataset}/{self.args.arch}/{self.args.name_method}_{self.args.trainset_name}.txt"
            # if os.path.isfile(name_file) == True:
            #     f = open(name_file, "a")
            #     print("Adding to already existing .txt file to avoid overwritting")
            # else:
            #     f = open(name_file, "w")
            # if self.args.same_query_size:
            #     type_n_query = "fixé"
            # else:
            #     type_n_query = "max"
            # f.write("\n\n")
            # f.write(f"lame {self.args.trainset_name},")
            # if self.args.number_tasks != "max":
            #     f.write(f"nb_tasks_{self.args.number_tasks},")
            # if self.args.s_use_all_train_set:
            #     s_sur_quoi = "sur train set entier"
            # else:
            #     s_sur_quoi = "sur support uniquement"
            # # f.write(f" sampling={self.args.sampling}, avec data augmentation, {self.args.covariance_used} ({s_sur_quoi}), n_query={self.args.n_query} ({type_n_query}), {self.args.normalisation_couleur} color normalization [patch_size={self.args.patch_size}" +
            # #         f", transform_size={self.args.transform_size}, random moyenné {nb_passages_pour_moyenner}x, pas de MinMaxScaler] \n\n")
            # if self.args.method == "paddle":
            #     f.write(
            #         f"methode={self.args.method}, sampling={self.args.sampling}, {self.args.normalisation_couleur} color normalization, transform_size={self.args.transform_size}, n_query={self.args.n_query} ({type_n_query}) \n[patch_size={self.args.patch_size}"
            #         + f", random moyenné {nb_passages_pour_moyenner}x, pas de MinMaxScaler, avec data augmentation, {self.args.covariance_used} ({s_sur_quoi})] \n"
            #         + f"select_support_nb_elements={self.args.select_support_nb_elements}, predire_une_seule_classe={self.args.predire_une_seule_classe}, gamma={self.args.gamma}\n\n"
            #     )

            #     f.write(
            #         "alpha        \t"
            #         + "metric       \t"
            #         + "\t".join([str(a) + "-shot  " for a in self.args.shots])
            #         + "\n"
            #     )
            # else:
            #     f.write(
            #         f"methode={self.args.method}, sampling={self.args.sampling}, {self.args.normalisation_couleur} color normalization, transform_size={self.args.transform_size}, n_query={self.args.n_query} ({type_n_query}) \n[patch_size={self.args.patch_size}"
            #         + f", random moyenné {nb_passages_pour_moyenner}x, pas de MinMaxScaler, avec data augmentation, {self.args.covariance_used} ({s_sur_quoi})] \n"
            #         + f"select_support_nb_elements={self.args.select_support_nb_elements}, predire_une_seule_classe={self.args.predire_une_seule_classe}\n\n"
            #     )

            #     f.write(
            #         "alpha        \t"
            #         + "metric       \t"
            #         + "\t".join([str(a) + "-shot  " for a in self.args.shots])
            #         + "\n"
            #     )

        if self.args.method != "paddle":
            self.args.alphas = [0]
            self.args.gamma = 1.0

        ## If we want to experiment for many values of alpha
        for alpha in self.args.alphas:
            self.args.alpha = alpha
            accuracies = {}
        
            ## if we want to do the experiment for many values of shots
            for n_shot in self.args.shots:
                nb_tasks_real = 0
                predictions_par_patch = {}
        
                global_truth = []
                global_prediction = []
                global_confidence = []
                global_patch_order = []

                # find lowest occurency
                #counter = Counter(extracted_features_dic_support["concat_labels"].tolist())
                #max_shots_not_augmented = min(counter.values())


                # create the support set once for each task
                x_support, y_support = generate_support_set(extracted_features_dic_support, extract_features_dic_support_only_augmented, n_shot)
    
                
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

                    ## Run task
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

                    ## Task completed for the sliding window
                    nb_tasks_real += 1

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

                ## Reconstruction of the predictions using majority vote
                ## construct each patch which prediction has been made, how many times
                ## finaly by majority voting reconstruct the predictions and display it
                ## saving the figure in results/predictions/trainset_name/predAnnotations/_alpha_{}_nshots_{}_covmatrix_{}_trainset_{supp_hospital}
                # If in prediction mode (e.g. if we don't have ground truth)
                if self.args.prediction:
                    for n in range(len(global_patch_order)):
                        patch = global_patch_order[n]
                        classe = global_prediction[n]
                        _, _, _, i, _, j = patch.split("_")
                        i, j = int(i), int(j)
                        if not (i, j) in predictions_par_patch:
                            predictions_par_patch[i, j] = [
                                0 for c in range(self.args.n_ways)
                            ]
                        predictions_par_patch[i, j][classe] += 1

                    display_prediction(
                        compo_querysets,
                        predictions_par_patch,
                        self.args,
                        alpha,
                        n_shot,
                        random_number,
                    )


                # create the confusion matrix and the classification report
                if self.args.evaluation and (not self.args.overlapping):
                    S_all = '' if not self.args.s_use_all_train_set else '_S_all'
                    save_dir = self.args.evaluation_dir + f'/{self.args.prefix}_{self.args.method}_{alpha}_{str(n_shot)}_shots{S_all}/'

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)

                    create_confusion_matrix(global_truth, global_prediction, save_dir)
                    with open(save_dir + "classification_report.txt", "w") as f:
                        f.write(classification_report(y_true=global_truth, y_pred=global_prediction, target_names=['P', 'H', 'Né', 'TL', 'Fi', 'T']))

                    accuracy = accuracy_score(global_truth, global_prediction)
                    accuracies[n_shot] = accuracy

                    print(f"Accuracy for {n_shot}-shot: {accuracy}")


                elif self.args.evaluation and self.args.overlapping:
                    S_all = '' if not self.args.s_use_all_train_set else '_S_all'
                    save_dir = self.args.evaluation_dir + f'/{self.args.prefix}_{self.args.method}_{alpha}_{str(n_shot)}_shots{S_all}_woverlapping/'
                    
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
                    accuracies[n_shot] = accuracy

                    print(f"Accuracy for {n_shot}-shot with overlapping: {accuracy}")

                else:
                    raise ValueError("To be implemented")

                    # # on calcule les métriques
                    # all_metrics_list["accuracy"].append(
                    #     accuracy_score(global_truth, global_prediction)
                    # )
                    # all_metrics_list["f1_micro"].append(
                    #     f1_score(global_truth, global_prediction, average="micro")
                    # )
                    # all_metrics_list["f1_macro"].append(
                    #     f1_score(global_truth, global_prediction, average="macro")
                    # )
                    # all_metrics_list["f1_weighted"].append(
                    #     f1_score(global_truth, global_prediction, average="weighted")
                    # )
                    # all_metrics_list["balanced_acc"].append(
                    #     balanced_accuracy_score(global_truth, global_prediction)
                    # )

                    # # on calcule la matrice de confusion
                    # cf_matrix = confusion_matrix(
                    #     global_truth,
                    #     global_prediction,
                    #     labels=list(range(self.args.n_ways)),
                    #     normalize="true",
                    # )

                    # # on crée l'image de la matrice de confusion
                    # legend = f"Matrice de confusion, {self.args.covariance_used}, alpha=n_query={self.args.n_query} ({type_n_query}), n_shots={shot}, {self.args.normalisation_couleur}_colornorm° ({nb_passages_pour_moyenner}*{nb_tasks_real//nb_passages_pour_moyenner} tasks)"
                    # if self.args.sampling in ["sliding_window", "squares"]:
                    #     filename = f"conf_matrix_iters_{self.args.iter}_lame_{self.args.trainset_name}_{self.args.sampling}_windowsize_{self.args.window_size}_alpha_{alpha}_nshots_{shot}_transformsize_{self.args.transform_size}_normalization_{self.args.normalisation_couleur}_covmatrix_{self.args.covariance_used}"
                    #     if self.args.covariance_used == "S_full":
                    #         if self.args.s_use_all_train_set:
                    #             filename += "_sur_alltrainset"
                    #         else:
                    #             filename += "_sur_justesupport"
                    # else:
                    #     filename = f"conf_matrix_{self.args.covariance_used}_nquery_{self.args.n_query}_samequerysize_{self.args.same_query_size}_patchsize_{self.args.patch_size}_transformsize_{self.args.transform_size}_colornorm_{self.args.normalisation_couleur}_nshot_{shot}_trainset_{self.args.support_hospital}"
                    # if self.args.number_tasks != "max":
                    #     filename = f"nb_tasks_{self.args.number_tasks}_" + filename
                    # if self.args.select_support_nb_elements != False:
                    #     filename += (
                    #         f"_select_support_{self.args.select_support_nb_elements}"
                    #     )
                    # if self.args.predire_une_seule_classe:
                    #     filename += "_predire_une_seule_classe"
                    # create_confusion_matrix(legend, filename, cf_matrix)
                    # #super_logger.info(f"Support : {patients_list}")

                    # histogrammes = False
                    # if histogrammes:
                    #     # On calcule et trace l'histogramme de confiance
                    #     list_classes = ["AM", "AN", "NT", "RE", "VE"]
                    #     une_figure = False
                    #     graph_log = False
                    #     if une_figure:
                    #         plt.figure(figsize=(12, 20))
                    #         fig, axs = plt.subplots(self.args.n_ways, 1, sharex=True)
                    #     for k in range(self.args.n_ways):
                    #         l = [
                    #             global_confidence[i]
                    #             for i in range(len(global_confidence))
                    #             if global_prediction[i] == k
                    #         ]
                    #         if une_figure:
                    #             axs[k].hist(l, bins=[x / 50 for x in range(51)])
                    #             axs[k].set_title(
                    #                 f"Confidence when predicted_class={list_classes[k]}",
                    #                 y=1.0,
                    #                 pad=-14,
                    #             )
                    #         else:
                    #             plt.figure(figsize=(12, 7))
                    #             if graph_log:
                    #                 plt.hist(
                    #                     l, bins=[x / 50 for x in range(51)], log=True
                    #                 )
                    #                 plt.savefig(
                    #                     f"histograms/hist_log_nshots_{shot}_predclass_{list_classes[k]}.png"
                    #                 )
                    #                 plt.close()
                    #             else:
                    #                 plt.hist(l, bins=[x / 50 for x in range(51)])
                    #                 plt.savefig(
                    #                     f"histograms/hist_linear_nshots_{shot}_predclass_{list_classes[k]}.png"
                    #                 )
                    #                 plt.close()
                    #     if une_figure:
                    #         plt.savefig(f"histograms/hist_nshots_{shot}.png")
                    #         plt.close()

            # if self.args.evaluation:
            #     print(accuracies)
            #     #mean_accuracies = np.asarray(accuracies.values).mean()
            #     #print(mean_accuracies)
            # else:
            #     mean_accuracies = "No accuracy here, this is a prediction"

            
            #save the accuracies dictionary into a csv file
            save_dir = self.args.evaluation_dir
            df = pd.DataFrame(list(accuracies.items()), columns=['n_shots', 'accuracy'])
            df.to_csv(self.args.evaluation_dir + f'{self.args.prefix}_{self.args.method}_{n_shot}_woverlap_{self.args.overlapping}_n_shots_analysis{S_all}.csv', index=False)


        logging.info("----- Test has ended -----")

        
            # if self.args.evaluation:
            #     path = "results/test/{}/{}".format(self.args.dataset, self.args.arch)
            #     # name_file = path + '/{}.txt'.format(self.args.name_method)
            #     name_file = f"results/test/{self.args.dataset}/{self.args.arch}/{self.args.name_method}_{self.args.trainset_name}.txt"
            #     if not os.path.exists(path):
            #         os.makedirs(path)
            #     if os.path.isfile(name_file) == True:
            #         f = open(name_file, "a")
            #         print("Adding to already existing .txt file to avoid overwritting")
            #     else:
            #         f = open(name_file, "w")

            #     for metric_name in all_metrics:
            #         f.write(
            #             str(param)
            #             + " " * (14 - len(str(param)))
            #             + "\t"
            #             + metric_name
            #             + " " * (12 - len(metric_name))
            #             + "\t"
            #         )
            #         for shot in self.args.shots:
            #             self.logger.info(
            #                 "{}-shot mean {} over {} tasks: {}".format(
            #                     shot,
            #                     metric_name,
            #                     self.args.number_tasks,
            #                     all_metrics_list[metric_name][
            #                         self.args.shots.index(shot)
            #                     ],
            #                 )
            #             )
            #             f.write(
            #                 str(
            #                     round(
            #                         all_metrics_list[metric_name][
            #                             self.args.shots.index(shot)
            #                         ],
            #                         3,
            #                     )
            #                 )
            #                 + "     \t"
            #             )
            #         f.write("\n")

        # if self.args.evaluation:
        #     f.close()

        # super_logger.info(
        #     "\t Color normalisation: {}".format(self.args.normalisation_couleur)
        # )
        # super_logger.info("\t Sampling: {}".format(self.args.sampling))
        # # super_logger.info(
        # #     "\t Features parameters: {}".format(
        # #         "__support_" + self.args.support_hospital
        # #     )
        # # )
        # super_logger.info(f"\t Method: {self.args.method}")
        # #super_logger.info(f"\t Support: {patients_list}")
        # super_logger.info(f"\t dataset: {self.args.dataset}")
        # super_logger.info(f"\t architecture: {self.args.arch}")
        # super_logger.info(f"\t method_name: {self.args.name_method}")
        # super_logger.info(f"\t trainset_name: {self.args.trainset_name}")
        # super_logger.info(f"\t dataset: {self.args.dataset}")
        # super_logger.info(
        #     f"\t alpha={self.args.alpha}, gamma={self.args.gamma}, shots={self.args.shots}"
        # )
        # super_logger.info(
        #     f"This is an informational message:  Test_{random_number} Finished! \n"
        # )
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
