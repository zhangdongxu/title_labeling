#!/bin/sh

#########################################################
#This script counts frequency without a title list given.
#This script merges and prunes after each input directory is counted.
#This script merges after all the input directories are counted, merged, and pruned.
##########################################################

modeltype=weightedwindow #paragraph, window or weightedwindow
modelname=descriptor-weightedwindow-weight2-onlyusebrackets #any name you like
process_num=6 #number of process
prune_threshold=1.0

workdir=/home/dongxu/work/title_labeling
inputdir=/zfs/octo/sogout/outputs/filtered
descriptor_path=/zfs/octp/sogout/outputs/extract_desc/desc.txt

count=0

mkdir -p $workdir/model/$modelname
mkdir -p $workdir/model/$modelname.prune

for partition in 0 2 4 6 8
do

mkdir -p $workdir/log/log.$modelname
mkdir -p $workdir/outputs/$modelname/freq/sogout_data.$partition/

for i in `seq -w 0000 0409`
do
  input=$inputdir/sogout_data.$partition/part-m-$i.bz2
  output=$workdir/outputs/$modelname/freq/sogout_data.$partition/part-m-$i

  echo "input:" $input
  echo "output:" $output 
  python3 descriptor.py -b --model_type $modeltype --window_size 10 --smooth_factor 2 --input $input --descriptor $descriptor_path --model $output > $workdir/log/log.$modelname/log.sogout_data.$partition.part-m-$i &

  count=$((count+1))
  if [ $count -ge $process_num ]; then
    wait
    count=0
  fi
done

wait
python3 descriptor.py -m --mergedir $workdir/outputs/$modelname/freq/sogout_data.$partition --model $workdir/model/$modelname/$modelname.sogout_data.$partition
python3 descriptor.py -p --prune_threshold $prune_threshold --model $workdir/model/$modelname/$modelname.sogout_data.$partition.p --prune_file $workdir/model/$modelname.prune/$modelname.sogout_data.$partition.prune

done

python3 descriptor.py -m --mergedir $workdir/model/$modelname.prune --model $workdir/model/$modelname.prune
