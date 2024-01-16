# BADM FMNIST
exec -a "BADM_420" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 2 --target_outlier_class 0 --gpu 2 --random_seed 0 &
exec -a "BADM_486" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 8 --target_outlier_class 6 --gpu 2 --random_seed 0 &
exec -a "BADM_440" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 4 --target_outlier_class 0 --gpu 2 --random_seed 0 &

exec -a "BADM_426" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 2 --target_outlier_class 6 --gpu 2 --random_seed 0 &
exec -a "BADM_406" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 0 --target_outlier_class 6 --gpu 2 --random_seed 0 &
exec -a "BADM_476" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 7 --target_outlier_class 6 --gpu 2 --random_seed 0 &

exec -a "BADM_460" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 6 --target_outlier_class 0 --gpu 2 --random_seed 0 &
exec -a "BADM_430" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 3 --target_outlier_class 0 --gpu 2 --random_seed 0 &
exec -a "BADM_480" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 8 --target_outlier_class 0 --gpu 2 --random_seed 0 &
exec -a "BADM_490" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 9 --target_outlier_class 0 --gpu 2 --random_seed 0 &

exec -a "BADM_442" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 4 --target_outlier_class 2 --gpu 2 --random_seed 0 &
exec -a "BADM_446" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 4 --target_outlier_class 6 --gpu 2 --random_seed 0 &
exec -a "BADM_443" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 4 --target_outlier_class 3 --gpu 2 --random_seed 0 &

wait

#BADM nb15
exec -a "BADM_nb15" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15 --dataset_name nb15 --gpu 0 --sample_count 1000 --random_seed 0 &

#BADM SQB
exec -a "BADM_SQB" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/SQB --dataset_name SQB --gpu 0 --sample_count 200 --random_seed 0 &

# nb15 上不同eta0测试
exec -a "BADM_nb15_eta0_2" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --sample_count 1000 --eta_0 2&
exec -a "BADM_nb15_eta0_5" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --sample_count 1000 --eta_0 5&
exec -a "BADM_nb15_eta0_10" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --sample_count 1000 --eta_0 10&
exec -a "BADM_nb15_eta0_20" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --sample_count 1000 --eta_0 20&
exec -a "BADM_nb15_eta0_30" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --sample_count 1000 --eta_0 30&
wait

exec -a "BADM_nb15_eta0_40" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --sample_count 1000 --eta_0 40&
exec -a "BADM_nb15_eta0_50" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --sample_count 1000 --eta_0 50&
exec -a "BADM_nb15_eta0_100" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --sample_count 1000 --eta_0 100&
exec -a "BADM_nb15_eta0_200" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --sample_count 1000 --eta_0 200&
exec -a "BADM_nb15_eta0_500" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --sample_count 1000 --eta_0 500&
wait

# contamination test
exec -a "BADM_nb15_contamination_2" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/contamination --dataset_name 1_Unlabelled_data --gpu 2 --sample_count 1000 --random_seed 0 &
exec -a "BADM_nb15_contamination_4" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/contamination --dataset_name 2_Unlabelled_data --gpu 2 --sample_count 1000 --random_seed 0 &
exec -a "BADM_nb15_contamination_6" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/contamination --dataset_name 3_Unlabelled_data --gpu 2 --sample_count 1000 --random_seed 0 &
exec -a "BADM_nb15_contamination_8" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/contamination --dataset_name 4_Unlabelled_data --gpu 2 --sample_count 1000 --random_seed 0 &
wait
exec -a "BADM_nb15_contamination_10" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/contamination --dataset_name 5_Unlabelled_data --gpu 2 --sample_count 1000 --random_seed 0 &
exec -a "BADM_nb15_contamination_12" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/contamination --dataset_name 6_Unlabelled_data --gpu 2 --sample_count 1000 --random_seed 0 &
exec -a "BADM_nb15_contamination_14" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/contamination --dataset_name 7_Unlabelled_data --gpu 2 --sample_count 1000 --random_seed 0 &
exec -a "BADM_nb15_contamination_16" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/contamination --dataset_name 8_Unlabelled_data --gpu 2 --sample_count 1000 --random_seed 0 &
wait

# BADM用于计算anchor的采样个数
exec -a "BADM_nb15_sample_count_500" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_sample_count --dataset_name nb15 --gpu 2 --sample_count 500 --random_seed 0 &
exec -a "BADM_nb15_sample_count_1000" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_sample_count --dataset_name nb15 --gpu 2 --sample_count 1000 --random_seed 0 &
exec -a "BADM_nb15_sample_count_1500" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_sample_count --dataset_name nb15 --gpu 2 --sample_count 1500 --random_seed 0 &
exec -a "BADM_nb15_sample_count_2000" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_sample_count --dataset_name nb15 --gpu 2 --sample_count 2000 --random_seed 0 &
exec -a "BADM_nb15_sample_count_2500" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_sample_count --dataset_name nb15 --gpu 2 --sample_count 2500 --random_seed 0 &
exec -a "BADM_nb15_sample_count_3000" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_sample_count --dataset_name nb15 --gpu 2 --sample_count 3000 --random_seed 0 &

wait

# BADM nb15 不同target类
exec -a "BADM_DoS" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_target_class_num --dataset_name nb15 --gpu 2 --sample_count 1000 --random_seed 0 --nb15_target_class DoS&
exec -a "BADM_Generic" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_target_class_num --dataset_name nb15 --gpu 2 --sample_count 1000 --random_seed 0 --nb15_target_class Generic&
exec -a "BADM_Backdoor" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_target_class_num --dataset_name nb15 --gpu 2 --sample_count 1000 --random_seed 0 --nb15_target_class Backdoor&
exec -a "BADM_DoS_Generic" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_target_class_num --dataset_name nb15 --gpu 2 --sample_count 1000 --random_seed 0 --nb15_target_class DoS Generic&
exec -a "BADM_DoS_Backdoor" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_target_class_num --dataset_name nb15 --gpu 2 --sample_count 1000 --random_seed 0 --nb15_target_class DoS Backdoor&
exec -a "BADM_Generic_Backdoor" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_target_class_num --dataset_name nb15 --gpu 2 --sample_count 1000 --random_seed 0 --nb15_target_class Generic Backdoor&

wait

# BADM 不同low和high数量
exec -a "BADM_nb15_1_1" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --sample_count 1000 --random_seed 0 --s_target 1&
exec -a "BADM_nb15_1_10" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --sample_count 1000 --random_seed 0 --s_target 10&
exec -a "BADM_nb15_1_50" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --sample_count 1000 --random_seed 0 --s_target 50&
exec -a "BADM_nb15_1_100" python main.py --model_type BiasedADM --dir_path ./result/BADM_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --sample_count 1000 --random_seed 0 --s_target 100&