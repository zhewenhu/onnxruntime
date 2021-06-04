#!/bin/bash 

while getopts p: parameter
do case "${parameter}"
in 
p) PERF_DIR=${OPTARG};;
esac
done 

id=$(sudo docker create ort-master)
sudo docker cp $id:/code/onnxruntime/build/Linux/Release/dist/ $PERF_DIR 
sudo docker rm -v $id
