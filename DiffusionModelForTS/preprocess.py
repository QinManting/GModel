# /usr/bin/env python
# -*- coding: utf-8 -*-

# @file: preprocess.py
# @time: 2026/04/25 
# @author: qin manting
# @version: 1.0
# @eamil: 2295547681@qq.com
# @desc: the preprocess script


import pandas as pd


def main():
	input_path = "./data/GS.xlsx"
	output_path = "./data/GS_min15.csv"

	df = pd.read_excel(input_path)
	df["Date"] = pd.to_datetime(df["Date"])

	filtered = df[df["Date"].dt.minute == 15].copy()
	result = filtered[["Date", "Prices", "新能源预测"]].rename(
		columns={"Date": "date","Prices": "price", "新能源预测": "generation"}
	)

	result.to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
	main()
