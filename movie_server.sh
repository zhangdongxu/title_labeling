#!/bin/sh
python movie_server.py --model model/model.p --topk 10 > log.server
