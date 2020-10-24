#!/bin/bash

python runtagger.py sents.test model-file sents.out && python eval.py sents.out sents.answer
