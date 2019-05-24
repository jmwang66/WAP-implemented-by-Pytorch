#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python -u ./translate.py -k 10 ./result/WAP_params.pkl \
	./data/dictionary.txt \
	./data/offline-test.pkl \
	./data/test_caption.txt \
	./result/test_decode_result.txt \
	./result/test.wer
