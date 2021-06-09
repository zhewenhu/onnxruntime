#!/bin/bash 

while getopts p: parameter
do case "${parameter}"
in 
p) PERF_DIR=${OPTARG};;
esac
done 

WHEEL_FOLDER=ort-trt-ep

cd $PERF_DIR
rm -rf $WHEEL_FOLDER
mkdir $WHEEL_FOLDER
id=$(sudo docker create ort-master)
sudo docker cp $id:/code/onnxruntime/build/Linux/Release/dist/ $PERF_DIR/$WHEEL_FOLDER
sudo docker rm -v $id
