#!/bin/sh
python3 movie_server.py --model model/model.p --topk 10 > log.server
