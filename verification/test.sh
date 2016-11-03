#!/usr/bin/env sh

#align_data_path=/home/robert/myCoding/suresecure/dataset/lfw_data/lfw_align_128x128_wu_net
#model_prefix=../model/lightened_cnn/lightened_cnn
#epoch=166
## evaluate on lfw
#python lfw.py --lfw-align $align_data_path --model-prefix $model_prefix --epoch $epoch



#align_data_path=/media/robert/746703E95E9EFB26/WorkSpace/suresecure/face_verification/lfw_align_lightened_cnn_mxnet
#model_prefix=../model/lightened_cnn_retrain/lightened_cnn
#epoch=200
## evaluate on lfw
#python lfw.py --lfw-align $align_data_path --model-prefix $model_prefix --epoch $epoch


filelist=/media/robert/746703E95E9EFB26/WorkSpace/suresecure/face_verification/lfw_align_lightened_cnn_mxnet/filepath.txt
#filelist=/home/robert/myCoding/suresecure/dataset/lfw_data/lfw_align_128x128_lightened_cnn/filepath_partly.txt
matfile=lightened_cnn_mxnet_166
model_prefix=../model/lightened_cnn/lightened_cnn
epoch=166
# evaluate on lfw
python lfw_feature.py --image-list $filelist --model-prefix $model_prefix --epoch $epoch --mat-file $matfile

filelist=/media/robert/746703E95E9EFB26/WorkSpace/suresecure/face_verification/lfw_align_lightened_cnn_mxnet/filepath.txt
#filelist=/home/robert/myCoding/suresecure/dataset/lfw_data/lfw_align_128x128_lightened_cnn/filepath_partly.txt
matfile=lightened_cnn_mxnet_retrain_200
model_prefix=../model/lightened_cnn_retrain/lightened_cnn
epoch=200
# evaluate on lfw
python lfw_feature.py --image-list $filelist --model-prefix $model_prefix --epoch $epoch --mat-file $matfile
