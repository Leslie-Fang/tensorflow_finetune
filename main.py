# -*- coding: utf-8 -*-
from test import test_fp32
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("mode", help="display a square of a given number",type=int)
	args = parser.parse_args()
	if args.mode is 1:
		#FP32
		test_fp32()


