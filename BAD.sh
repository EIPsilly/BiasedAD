# BAD FMNIST
exec -a "BAD_420" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 2 --target_outlier_class 0 --gpu 2 --random_seed 0&

exec -a "BAD_426" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 2 --target_outlier_class 6 --gpu 2 --random_seed 0&
exec -a "BAD_406" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 0 --target_outlier_class 6 --gpu 2 --random_seed 0&
exec -a "BAD_486" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 8 --target_outlier_class 6 --gpu 2 --random_seed 0&
exec -a "BAD_476" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 7 --target_outlier_class 6 --gpu 2 --random_seed 0&

exec -a "BAD_460" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 6 --target_outlier_class 0 --gpu 2 --random_seed 0&
exec -a "BAD_430" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 3 --target_outlier_class 0 --gpu 2 --random_seed 0&
exec -a "BAD_480" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 8 --target_outlier_class 0 --gpu 2 --random_seed 0&
exec -a "BAD_490" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 9 --target_outlier_class 0 --gpu 2 --random_seed 0&

#BAD nb15
exec -a "BAD_nb15" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15 --dataset_name nb15 --gpu 2 --random_seed 0&

#BAD SQB
exec -a "BAD_SQB" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/SQB --dataset_name SQB --gpu 2 --random_seed 0 &

wait

# contamination Figure 6
exec -a "BAD_nb15_contamination_2" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/contamination --dataset_name 1_Unlabelled_data --gpu 2 --random_seed 0&
exec -a "BAD_nb15_contamination_4" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/contamination --dataset_name 2_Unlabelled_data --gpu 2 --random_seed 0&
exec -a "BAD_nb15_contamination_6" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/contamination --dataset_name 3_Unlabelled_data --gpu 2 --random_seed 0&
exec -a "BAD_nb15_contamination_8" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/contamination --dataset_name 4_Unlabelled_data --gpu 2 --random_seed 0&
exec -a "BAD_nb15_contamination_10" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/contamination --dataset_name 5_Unlabelled_data --gpu 2 --random_seed 0&
exec -a "BAD_nb15_contamination_12" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/contamination --dataset_name 6_Unlabelled_data --gpu 2 --random_seed 0&
exec -a "BAD_nb15_contamination_14" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/contamination --dataset_name 7_Unlabelled_data --gpu 2 --random_seed 0&
wait

# BAD nb15 不同target类
# exec -a "BAD_DoS" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_target_class_num --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_target_class DoS&
# exec -a "BAD_Generic" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_target_class_num --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_target_class Generic&
# exec -a "BAD_Backdoor" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_target_class_num --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_target_class Backdoor&
# exec -a "BAD_DoS_Generic" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_target_class_num --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_target_class DoS Generic&
# exec -a "BAD_DoS_Backdoor" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_target_class_num --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_target_class DoS Backdoor&
# exec -a "BAD_Generic_Backdoor" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_target_class_num --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_target_class Generic Backdoor&

# wait


# BAD 不同target数量 Figure5 A 3*1 3*10 3*50 3*100 Figure5 A
# BAD 不同low和high数量 Figure5 B
exec -a "BAD_nb15_1_1" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --random_seed 0 --s_non_target 1 --s_target 1&
exec -a "BAD_nb15_1_10" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --random_seed 0 --s_non_target 1 --s_target 10&
exec -a "BAD_nb15_1_50" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --random_seed 0 --s_non_target 1 --s_target 50&
exec -a "BAD_nb15_1_100" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --random_seed 0 --s_non_target 1 --s_target 100&

exec -a "BAD_nb15_10_1" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --random_seed 0 --s_non_target 10 --s_target 1&
exec -a "BAD_nb15_10_10" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --random_seed 0 --s_non_target 10 --s_target 10&
wait
exec -a "BAD_nb15_10_50" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --random_seed 0 --s_non_target 10 --s_target 50&
exec -a "BAD_nb15_10_100" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --random_seed 0 --s_non_target 10 --s_target 100&

exec -a "BAD_nb15_50_1" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --random_seed 0 --s_non_target 50 --s_target 1&
exec -a "BAD_nb15_50_10" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --random_seed 0 --s_non_target 50 --s_target 10&
exec -a "BAD_nb15_50_50" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --random_seed 0 --s_non_target 50 --s_target 50&
exec -a "BAD_nb15_50_100" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --random_seed 0 --s_non_target 50 --s_target 100&
wait

exec -a "BAD_nb15_100_1" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --random_seed 0 --s_non_target 100 --s_target 1&
exec -a "BAD_nb15_100_10" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --random_seed 0 --s_non_target 100 --s_target 10&
exec -a "BAD_nb15_100_50" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --random_seed 0 --s_non_target 100 --s_target 50&
exec -a "BAD_nb15_100_100" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low_high --dataset_name nb15 --gpu 2 --random_seed 0 --s_non_target 100 --s_target 100&

# non-target 个数 和 class数量 Figure5 C
exec -a "BAD_nb15_1_1" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_non_target_class_num 1 --s_non_target 1&
exec -a "BAD_nb15_2_1" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_non_target_class_num 2 --s_non_target 1&
wait
exec -a "BAD_nb15_3_1" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_non_target_class_num 3 --s_non_target 1&
exec -a "BAD_nb15_4_1" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_non_target_class_num 4 --s_non_target 1&

exec -a "BAD_nb15_1_10" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_non_target_class_num 1 --s_non_target 10&
exec -a "BAD_nb15_2_10" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_non_target_class_num 2 --s_non_target 10&
exec -a "BAD_nb15_3_10" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_non_target_class_num 3 --s_non_target 10&
exec -a "BAD_nb15_4_10" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_non_target_class_num 4 --s_non_target 10&
wait

exec -a "BAD_nb15_1_50" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_non_target_class_num 1 --s_non_target 50&
exec -a "BAD_nb15_2_50" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_non_target_class_num 2 --s_non_target 50&
exec -a "BAD_nb15_3_50" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_non_target_class_num 3 --s_non_target 50&
exec -a "BAD_nb15_4_50" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_non_target_class_num 4 --s_non_target 50&

exec -a "BAD_nb15_1_100" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_non_target_class_num 1 --s_non_target 100&
exec -a "BAD_nb15_2_100" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_non_target_class_num 2 --s_non_target 100&
exec -a "BAD_nb15_3_100" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_low --dataset_name nb15 --gpu 2 --random_seed 0 --nb15_non_target_class_num 3 --s_non_target 100&
wait

# 不同eta0测试 Figure8 A
exec -a "BAD_nb15_eta0_2" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --eta_0 2&
exec -a "BAD_nb15_eta0_5" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --eta_0 5&
exec -a "BAD_nb15_eta0_10" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --eta_0 10&
exec -a "BAD_nb15_eta0_20" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --eta_0 20&
exec -a "BAD_nb15_eta0_30" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --eta_0 30&
wait

exec -a "BAD_nb15_eta0_40" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --eta_0 40&
exec -a "BAD_nb15_eta0_50" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --eta_0 50&
exec -a "BAD_nb15_eta0_100" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --eta_0 100&
exec -a "BAD_nb15_eta0_200" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --eta_0 200&
exec -a "BAD_nb15_eta0_500" python main.py --model_type BiasedAD --dir_path ./result/BAD_20231118/nb15_eta0 --dataset_name nb15 --gpu 2 --random_seed 0 --eta_0 500&

