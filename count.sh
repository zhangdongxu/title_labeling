#!/bin/sh

modeltype=window #paragraph, window or weightedwindow
modelname=descriptor-window #any name you like
process_num=1 #number of process

workdir=/home/dongxu/title_labeling
inputdir=/zfs/octp/sogout/outputs/filtered
descriptor_path=/zfs/octp/sogout/outputs/extract_desc/desc.txt
title_path=/zfs/octp/sogout/outputs/extract_desc/movie_titles_logp.txt
count=0

for partition in 1 3 5 7 9
do

mkdir -p $workdir/log/log.$modelname
mkdir -p $workdir/outputs/$modelname/freq/sogout_data.$partition/

for i in `seq -w 0000 0409`
do
  input=$inputdir/sogout_data.$partition/part-m-$i.bz2
  output=$workdir/outputs/$modelname/freq/sogout_data.$partition/part-m-$i

  echo "input:" $input
  echo "output:" $output 
  python descriptor.py -b --model_type $modeltype --window_size 10 --smooth_factor 2 --input $input --descriptor $descriptor_path --title $title_path --model $output > $workdir/log/log.$modelname/log.sogout_data.$partition.part-m-$i &

  count=$((count+1))
  if [ $count -eq $process_num ]; then
    wait
    count=0
  fi
done
done

mkdir -p $workdir/model
python descriptor.py -m --mergedir $workdir/outputs/$modelname --model $workdir/model/$modelname
