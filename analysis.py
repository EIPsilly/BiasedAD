import click
import string
import numpy as np
import pandas as pd
import os
import re

@click.command()
@click.argument("dir_path", type = str)
def main(dir_path):
    path = dir_path

    files = os.listdir(path)
    s = []
    res = []
    col_num = 0

    for file in files: #遍历文件夹
        data = []
        AUC = []
        PRC = []
        if not os.path.isdir(file):
            if not ("=" in file):
                continue
            if file.endswith(".npz") or file.endswith(".pdf"):
                continue
            f = open(path+"/"+file) #打开文件
            iter_f = iter(f) #创建迭代器
            count_num = 0
            for num, line in enumerate(f):
                data.append(line.split('|'))
                count_num = count_num + 1
            record_cnt = 0
            for foo in data:
                AUC.append(float(foo[0].split(': ')[1].split('%')[0]))
                PRC.append(float(foo[1].split(': ')[1].split('%')[0]))
                record_cnt += 1
            
            para_list = re.split(",|\.txt",file)[:-1]
            if (len(para_list) > col_num):
                col_num = len(para_list)
                col_name = list(map(lambda x: re.split("=",x)[0],para_list))
            value = list(map(lambda x: re.split("=",x)[1],para_list))
            res.append(value + [record_cnt, "{:.2f}%({:.2f})".format(np.mean(AUC), np.std(AUC)),'{:.2f}%({:.2f})'.format(np.mean(PRC), np.std(PRC))])
            
    df = pd.DataFrame(res, columns = col_name + ["record","auroc","auprc"],dtype=float)
    # print(df.sort_values(by=["target","non_target"]))
    print(df)
    path = "./analysis_result"
    # df.to_csv(path + "/{}-{}.csv".format("MRBAD", dir_path.replace("/","-")))
    # df.to_excel(path + "/{}-{}.xlsx".format("MRBAD", dir_path.replace("/","-")))


if __name__ == "__main__":
    # main(["nb15_BADM"])
    # main(["check/SQB_BADM"])
    main(["SQB_BADM"])