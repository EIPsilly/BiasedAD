import pandas as pd
import numpy as np
import logging
import random
import time
import json
from collections import Counter
import matplotlib.pyplot as plt

import click
from utils.write2txt import writer2txt
import os
import re
from BiasedAD import BiasedAD

@click.command()
@click.option('--model_type', type=click.Choice(["BiasedAD", "BiasedADM"]), default="BiasedAD")
@click.option('--dir_path', type=str, default="./result")
@click.option('--dataset_name', type=str)
@click.option('--device', type=str, default="cuda")
@click.option('--gpu', type=str, default="3")
@click.option('--intermediate_flag', type=bool, default=False)
@click.option("--random_seed" , type=int, default = None)

@click.option("--lr" , type=float, default = None)
@click.option("--epoch" , type=int, default = None)
@click.option("--batch_size" , type=int, default = None)
@click.option("--weight_decay" , type=float, default = 5e-7)

@click.option("--ae_lr" , type=float, default = None)
@click.option("--ae_epoch" , type=int, default = None)
@click.option("--ae_batch_size" , type=int, default = None)
@click.option("--ae_weight_decay" , type=float, default = 1e-6)
@click.option("--sample_count" , type=int, default = 100)

@click.option('--times', type=int, default=10)
@click.option('--eta_0', type=int, default=None)
@click.option('--eta_1', type=int, default=None)
@click.option('--eta_2', type=int, default=None)

# The follow three options are useful when the dataset is the fashionMNIST dataset
@click.option("--normal_class" , type=int, default = 4)
@click.option("--non_target_outlier_class" , type=int, default = 0)
@click.option("--target_outlier_class" , type=int, default = 6)

# The follow three options are useful when the dataset is the nb15 dataset with a fixed contamination ratio of 2%.
@click.option("--s_non_target" , type=int, default = 100)
@click.option("--s_target" , type=int, default = 100)
# Controls the number of non-target categories
@click.option("--nb15_non_target_class_num" , type=int, default = 4)

def main(model_type, dir_path, dataset_name, device, gpu, intermediate_flag, random_seed,
         lr, epoch, batch_size, weight_decay,
         ae_lr, ae_epoch, ae_batch_size, ae_weight_decay, sample_count,
         times, eta_0, eta_1, eta_2,
         s_non_target, s_target, nb15_non_target_class_num,
         normal_class, non_target_outlier_class, target_outlier_class):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    file_save_path = dir_path

    for i in range(times):
        contaminationRate = 2

        #The contaminated dataset name of nb-15 is "7_Unlabelled_data," where the number 7 indicates a contamination ratio of 7*2=14%.
        if re.search("Unlabelled_data", dataset_name):
            file_name = "dataset=NB15_contamination"

            if model_type == "BiasedAD":
                from datasets.nb15_contamination import NB15_contamination_Dataset
                dataset = NB15_contamination_Dataset(dataset_name, random_seed)
                default_eta_0 = 20
                default_eta_1 = 1
                default_eta_2 = 2
            elif model_type == "BiasedADM":
                from datasets.nb15_contamination_for_BADM import NB15_contamination_for_BADM
                dataset = NB15_contamination_for_BADM(dataset_name, s_non_target = 0, s_target = s_target, nb15_non_target_class_num = 4, seed = random_seed)
                default_eta_0 = 20
                default_eta_1 = 2
                default_eta_2 = 1
            
            file_name += ","
            file_name += "s_non_target=" + str(s_non_target) + ",s_target=" + str(s_target) + ",nb15_non_target_class_num=" +str(nb15_non_target_class_num) + ","
            net_name = 'mlp_for_nb15'
            contaminationRate = int(dataset_name.split("_")[0]) * 2
            ae_lr = 0.0001 if ae_lr is None else ae_lr
            ae_batch_size = 128 if ae_batch_size is None else ae_batch_size
            ae_epoch = 30 if ae_epoch is None else ae_epoch
            MRBAD_lr = 0.00001
            MRBAD_batch_size = 128
            MRBAD_epoch = 50


        elif dataset_name == "nb15":
            if model_type == "BiasedAD":
                from datasets.nb15 import NB15Dataset
                dataset = NB15Dataset(s_non_target, s_target, nb15_non_target_class_num, random_seed)
                default_eta_0 = 20
                default_eta_1 = 1
                default_eta_2 = 2
            elif model_type == "BiasedADM":
                from datasets.nb15_for_BADM import nb15_for_BADM
                dataset = nb15_for_BADM(0, s_target, nb15_non_target_class_num, random_seed)
                default_eta_0 = 20
                default_eta_1 = 2
                default_eta_2 = 1
            net_name = 'mlp_for_nb15'
            file_name = "s_non_target=" + str(s_non_target) + ",s_target=" + str(s_target) + ",nb15_non_target_class_num=" +str(nb15_non_target_class_num) + ","
            ae_lr = 0.0001 if ae_lr is None else ae_lr
            ae_batch_size = 128 if ae_batch_size is None else ae_batch_size
            ae_epoch = 30 if ae_epoch is None else ae_epoch
            MRBAD_lr = 0.00001
            MRBAD_batch_size = 128
            MRBAD_epoch = 50

        elif dataset_name == "fashionmnist":
            if normal_class != non_target_outlier_class:
                unlabeled_normal_number = 5000
                labeled_normal_number = 0
                test_normal = 1000

                unlabeled_non_target_outlier_number = 50
                labeled_non_target_outlier_number = 100
                test_non_target_outlier = 100

                unlabeled_target_outlier_number = 50
                labeled_target_outlier_number = 100
                test_target_outlier = 100
            else:
                unlabeled_normal_number = 5000
                labeled_normal_number = 0
                test_normal = 1000

                unlabeled_non_target_outlier_number = 0
                labeled_non_target_outlier_number = 0
                test_non_target_outlier = 0

                unlabeled_target_outlier_number = 100
                labeled_target_outlier_number = 100
                test_target_outlier = 100
            if model_type == "BiasedADM":
                labeled_non_target_outlier_number = 0

            if model_type == "BiasedAD":
                from datasets.fashionmnist import FashionMNIST_Dataset
                dataset = FashionMNIST_Dataset("./data",
                                            normal_class,
                                            unlabeled_normal_number,
                                            labeled_normal_number,
                                            test_normal,

                                            non_target_outlier_class,
                                            unlabeled_non_target_outlier_number,
                                            labeled_non_target_outlier_number,
                                            test_non_target_outlier,

                                            target_outlier_class,
                                            unlabeled_target_outlier_number,
                                            labeled_target_outlier_number,
                                            test_target_outlier,
                                            
                                            random_seed)
                default_eta_0 = 1
                default_eta_1 = 1
                default_eta_2 = 2
            elif model_type == "BiasedADM":
                from datasets.fashionmnist_for_BADM import fashionmnist_for_BADM
                dataset = fashionmnist_for_BADM("./data",
                                                normal_class,
                                                unlabeled_normal_number,
                                                labeled_normal_number,
                                                test_normal,

                                                non_target_outlier_class,
                                                unlabeled_non_target_outlier_number,
                                                labeled_non_target_outlier_number,
                                                test_non_target_outlier,
                                                
                                                target_outlier_class,
                                                unlabeled_target_outlier_number,
                                                labeled_target_outlier_number,
                                                test_target_outlier,

                                                random_seed)
                default_eta_0 = 1
                default_eta_1 = 1
                default_eta_2 = 2

            net_name = "fmnist_LeNet"
            if normal_class == non_target_outlier_class:
                file_save_path = dir_path + "/" + str(normal_class) + str(non_target_outlier_class) + "x"
            else:
                file_save_path = dir_path + "/" + str(normal_class) + "x" + str(target_outlier_class)

            file_name = "normal=" + str(normal_class) + ",non_target=" + str(non_target_outlier_class) + ",target=" + str(target_outlier_class) + ","
            ae_lr = 0.0001 if ae_lr is None else ae_lr
            ae_batch_size = 128 if ae_batch_size is None else ae_batch_size
            ae_epoch = 10 if ae_epoch is None else ae_epoch
            MRBAD_lr = 0.0001
            MRBAD_batch_size = 128
            MRBAD_epoch = 30

        elif dataset_name == "SQB":
            if model_type == "BiasedAD":
                from datasets.sqb import SQBDataset
                dataset = SQBDataset()
                default_eta_0 = 1
                default_eta_1 = 1
                default_eta_2 = 2
            elif model_type == "BiasedADM":
                from datasets.sqb_for_BADM import sqb_for_BADM
                dataset = sqb_for_BADM()
                default_eta_0 = 10
                default_eta_1 = 1
                default_eta_2 = 2

            file_name = "dataset=sqb,"
            net_name = "mlp_for_sqb"
            lr, epoch, batch_size
            ae_lr = 0.0001 if ae_lr is None else ae_lr
            ae_batch_size = 128 if ae_batch_size is None else ae_batch_size
            ae_epoch = 30 if ae_epoch is None else ae_epoch
            MRBAD_lr = 0.0001
            MRBAD_batch_size = 128
            MRBAD_epoch = 50
            
        if not (lr is None):
            MRBAD_lr = lr
        if not (epoch is None):
            MRBAD_epoch = epoch
        if not (batch_size is None):
            MRBAD_batch_size = batch_size
        if eta_0 is None:
            eta_0 = default_eta_0
        if eta_1 is None:
            eta_1 = default_eta_1
        if eta_2 is None:
            eta_2 = default_eta_2

        if not os.path.exists(file_save_path):
            os.makedirs(file_save_path)
        
        if not os.path.exists("./log/{}_log".format(dir_path.split("/")[-1])):
            os.makedirs("./log/{}_log".format(dir_path.split("/")[-1]))

        writer = writer2txt()
    
        model = BiasedAD(eta_0, eta_1, eta_2, model_type)
        model.set_network(net_name)
        model.pretrain(dataset, optimizer_name='adam',
                        lr=ae_lr,
                        n_epochs=ae_epoch,
                        batch_size=ae_batch_size,
                        weight_decay = ae_weight_decay,
                        device=device,
                        n_jobs_dataloader=0)
        if intermediate_flag:
            # save the characterization obtained after AE pretrain
            train_loader, test_loader = dataset.loaders(batch_size=128, drop_last_train = False)
            train_data_input, train_data_label, train_data_semi_target = model.intermediate_result(train_loader)
            test_data_input, test_data_label, test_data_semi_target = model.intermediate_result(test_loader)
            np.savez(file_save_path + "/" + file_name + "train_data.npz", train_data_input = train_data_input, train_data_label = train_data_label, train_data_semi_target = train_data_semi_target)
            np.savez(file_save_path + "/" + file_name + "test_data.npz", test_data_input = test_data_input, test_data_label = test_data_label, test_data_semi_target = test_data_semi_target)
        else:
            output_name = file_name + 'contaminationRate={},eta0={},eta1={},eta2={},MRBAD_lr={},MRBAD_batchsize={},MRBAD_epoch={},sample_count={},model_type={}'.format(str(contaminationRate), str(eta_0), str(eta_1), str(eta_2), str(MRBAD_lr), str(MRBAD_batch_size), str(MRBAD_epoch), str(sample_count), model_type)
            writer.set_output_name(output_name)
            writer.set_file_save_path(file_save_path)
            writer.set_path("{}/{}.txt".format(file_save_path, output_name), "./log/{}_log/{}".format(dir_path.split("/")[-1], output_name))
            model.train(dataset,
                        optimizer_name='adam',
                        lr=MRBAD_lr,
                        n_epochs=MRBAD_epoch,
                        batch_size=MRBAD_batch_size,
                        weight_decay=weight_decay,
                        device=device,
                        n_jobs_dataloader=0,
                        sample_count=sample_count)
            model.test(dataset, device=device, n_jobs_dataloader=0)

if __name__ == "__main__":
    main()