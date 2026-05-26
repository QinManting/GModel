# /usr/bin/env python
# -*- coding: utf-8 -*-

# @file: preprocess.py
# @time: 2026/04/25 
# @author: qin manting
# @version: 1.0
# @eamil: 2295547681@qq.com
# @desc: the preprocess script


import pandas as pd


def preprocess_data():
	input_path = "./data/GS.xlsx"
	output_path = "./data/GS_min00_2024.csv"

	df = pd.read_excel(input_path)
	df["Date"] = pd.to_datetime(df["Date"])

	df = df[(df["Date"].dt.year == 2024)]  # Filter for 2024

	filtered = df[df["Date"].dt.minute == 0].copy()
	result = filtered[["Date", "Prices", "新能源预测"]].rename(
		columns={"Date": "date","Prices": "price", "新能源预测": "generation"}
	)
	
	# 123月添加season=0, 456月添加season=1, 789月添加season=2, 101112月添加season=3
	result["season"] = result["date"].dt.month.apply(lambda x: (x-1)//3)
 
	result.to_csv(output_path, index=False, encoding="utf-8-sig")

def concat_data():
    # 将两个文件的数据合并成一个文件
	df1 = pd.read_csv("./data/GS_min00_2024.csv")
	df2 = pd.read_csv("./data/GS_min15_2024.csv")
	df3 = pd.read_csv("./data/GS_min30_2024.csv")
	df4 = pd.read_csv("./data/GS_min45_2024.csv")
	
	# 合并数据
	df = pd.concat([df1, df2, df3, df4], axis=0)
 
	# 重置索引
	df.reset_index(drop=True, inplace=True)
 
	# 保存合并后的数据
	df.to_csv("./data/GS_2024.csv", index=False, encoding="utf-8-sig")
 
def sort_data():
    # 读取数据
    df = pd.read_csv("./data/GS_2024.csv", encoding="utf-8-sig")
    
    # 只按照日期排序，时间不排序
	# 将日期列转换为datetime，按日期（天）排序，且在同一天内按分钟顺序排序
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["_day"] = df["date"].dt.date
        df["_minute"] = df["date"].dt.minute
        df.sort_values(["_day", "_minute"], inplace=True)
        df.drop(columns=["_day", "_minute"], inplace=True)
    else:
        # 如果没有 date 列，尝试解析第一列为日期时间并排序
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col])
        df["_day"] = df[first_col].dt.date
        df["_minute"] = df[first_col].dt.minute
        df.sort_values(["_day", "_minute"], inplace=True)
        df.drop(columns=["_day", "_minute"], inplace=True)
    # 保存排序后的数据
    df.to_csv("./data/GS_2024_sorted.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
	# preprocess_data()
	# concat_data()
	sort_data()
