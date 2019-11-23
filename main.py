# -*- coding: utf-8 -*-
from train import train
from test import test
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("mode", help="display a square of a given number",type=int)
	args = parser.parse_args()
	if args.mode is 0:
		train()
	elif args.mode is 1:
		#FP32
		test(0)
	elif args.mode is 2:
		#FP32 new model after fine tune
		test(1)

