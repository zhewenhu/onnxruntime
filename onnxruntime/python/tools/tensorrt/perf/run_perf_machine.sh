#!/bin/bash

# Parse Arguments
while getopts o:m: parameter
do case "${parameter}"
in 
o) OPTION=${OPTARG};;
m) MODEL_PATH=${OPTARG};;
e) EP_LIST=${OPTARG};;
esac
done 

# Variables
PERF_DIR=/home/hcsuser/perf/
WORKSPACE=/home/hcsuser/
#TRTEXEC="$(find ~/ -name trtexec | grep /bin/trtexec)"

# Select models to be tested or run selected-models 
if [ $OPTION == "onnx-zoo-models" ]
then 
    MODEL_PATH='model_list.json'
fi 

if [ $OPTION == "many-models" ]
then 
    MODEL_PATH=/home/hcsuser/mount/many-models
fi 

if [ $OPTION == "partner-models" ]
then 
    MODEL_PATH='partner_model_list.json'
fi

./perf.sh -d $PERF_DIR -o $OPTION -m $MODEL_PATH -w $WORKSPACE -e $EP_LIST
