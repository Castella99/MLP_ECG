#!/bin/bash

####################################
## Base Arguments Setting
####################################
instance_prefix=ecg_04

epoch=100
test_batch_size=1

lead_size=5000
lead_ch=12

####################################
## Fixed Arguments 
####################################

model_list="ECGNET_CNN_M1"
version="MetaPseudoLabel"
base_data_name=ecg_dataset_no_preprocessing.hdf5
DATE=$(date +"%Y_%m_%d_%H_%M")

for mn in ${model_list}
do
  for batch_size in 64
  do
    for cn in 2
    do
      for kfold in 5
      do
        for lr in 0.001
        do
          model=${mn}
          data_name=${base_data_name}
          learning_rate=${lr}
          if [ "$cn" -eq "2" ]; then
              class_name=ecg_1
          else
              class_name=ecg_0,ecg_1,ecg_2
          fi
          ####################################
          ## Aguments Merge
          ####################################

          instance=${instance_prefix}_${model}_ep${epoch}_bt${batch_size}_lr${learning_rate}_${data_name}_${kfold}_${version}_${DATE}

          ####################################
          ## Shell Execute
          ####################################
          train_ags="--name mpl -i ${instance} -m ${model} -d ${data_name} -c ${class_name} -bt ${batch_size} -lr ${learning_rate} -ep ${epoch} -ldsz ${lead_size} -ldch ${lead_ch} -kfold ${kfold}"

          ## Shell 
          echo ">> 1.Train Python Execute"
          echo "python -u ${train_ags} > ./logs/train_${instance}.log"

          python -u 10foldcv.py ${train_ags} > ./logs/cv_${instance}.log

        done
      done
    done
  done
done
