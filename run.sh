#! /bin/bash
set -exo pipefail

# set Default values
PREDICT=false
EPOCHS=60
LOSS=HybridLoss
BATCH_SIZE=3
ACTIVATION=leaky_relu
DROPOUT_RATE=0.0

while getopts d:t:n:b:l:s:a:r:e:p: flag
do
    case "${flag}" in
        d) DATA_PATH=${OPTARG};;
        t) MODEL_TYPE=${OPTARG};;
        n) MODEL_NAME=${OPTARG};;
        b) BACKBONE=${OPTARG};;
        l) LOSS=${OPTARG};;
        s) BATCH_SIZE=${OPTARG};;
        a) ACTIVATION=${OPTARG};;
        r) DROPOUT_RATE=${OPTARG};;
        e) EPOCHS=${OPTARG};;
        p) PREDICT=${OPTARG};;
    esac
done

MODEL=$MODEL_TYPE/$MODEL_NAME

# train model
python3 train_model.py --data_path $DATA_PATH --model_type $MODEL_TYPE --model_name $MODEL_NAME \
--backbone $BACKBONE --loss $LOSS --batch_size $BATCH_SIZE --activation $ACTIVATION --dropout $DROPOUT_RATE --epochs $EPOCHS

mkdir -p -m=776 Evaluation_logs/$MODEL_TYPE

#Evaluate model and save results in eval/MODEL_NAME.txt file
python3 evaluate_model.py --data_path $DATA_PATH --model_type $MODEL_TYPE --model_name $MODEL_NAME \
--backbone $BACKBONE --loss $LOSS >> Evaluation_logs/$MODEL_TYPE/$MODEL_NAME.txt

if [ $PREDICT = 'true' ]; then
    # make predictions with the validation set and convert them to rgb
    python3 create_predictions.py --data_path $DATA_PATH --model_type $MODEL_TYPE --model_name $MODEL_NAME --backbone $BACKBONE --split "val"
    python3 convert2rgb.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --split "val"

    # make predictions with the test set and convert them to rgb
    python3 create_predictions.py --data_path $DATA_PATH --model_type $MODEL_TYPE --model_name $MODEL_NAME --backbone $BACKBONE --split "test"
    python3 convert2rgb.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --split "test"

    # zip the generated images and place the compressed file into the archives folder
    zip -r archives/$MODEL_TYPE-$MODEL_NAME.zip predictions/$MODEL Evaluation_logs/$MODEL.txt Confusion_matrix/$MODEL.png
fi