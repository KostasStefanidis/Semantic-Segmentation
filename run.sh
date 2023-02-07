#! /bin/bash
set -e
set pipefail
set -x

numArg=$#
HELP=false

# set Default values
BACKBONE=None
LOSS=HybridLoss
BATCH_SIZE=3
ACTIVATION=leaky_relu
DROPOUT_RATE=0.0
AUGMENT=false
EPOCHS=60
PREDICT=false

for ((i=1 ; i <= $numArg ; i++))
do
    if [ "$1" == "-h" ] || [ "$1" == "--help" ]
        then
            HELP=true
        fi

    if [ "$1" == "-d" ] || [ "$1" == "--data-path" ]
        then
            DATA_PATH=$2 #mandatory
        fi

    if [ "$1" == "-t" ] || [ "$1" == "--model-type" ]
        then
            MODEL_TYPE=$2 #mandatory
        fi

    if [ "$1" == "-n" ] || [ "$1" == "--model-name" ]
        then
            MODEL_NAME=$2 #mandatory
        fi

    if [ "$1" == "-b" ] || [ "$1" == "--backbone" ]
        then
            BACKBONE=$2
        fi

    if [ "$1" == "-l" ] || [ "$1" == "--loss" ]
        then
            LOSS=$2
        fi

    if [ "$1" == "-batch-size" ] || [ "$1" == "--batch-size" ]
        then
            BATCH_SIZE=$2
        fi

    if [ "$1" == "-activation" ] || [ "$1" == "--activation" ]
        then
            ACTIVATION=$2
        fi

    if [ "$1" == "-dropout" ] || [ "$1" == "--dropout" ]
        then
            DROPOUT_RATE=$2
        fi

    if [ "$1" == "-a" ] || [ "$1" == "--augment" ]
        then
            AUGMENT=true
        fi

    if [ "$1" == "-e" ] || [ "$1" == "--epochs" ]
        then
            EPOCHS=$2
        fi

    if [ "$1" == "-p" ] || [ "$1" == "--predict" ]
        then
            PREDICT=true
        fi

    #shift 'consumes' the argument and then the loop can move to the next one
    shift
done

MODEL=$MODEL_TYPE/$MODEL_NAME

echo $HELP
echo $DATA_PATH
echo $MODEL_TYPE
echo $MODEL_NAME
echo $BACKBONE
echo $LOSS
echo $BATCH_SIZE
echo $ACTIVATION
echo $DROPOUT_RATE
echo $AUGMENT
echo $EPOCHS
echo $PREDICT

displayHelp(){
    echo "HELP"
    # TODO
}

main(){
    # train model
    python3 train_model.py --data_path $DATA_PATH --model_type $MODEL_TYPE --model_name $MODEL_NAME --backbone $BACKBONE\
     --loss $LOSS --batch_size $BATCH_SIZE --activation $ACTIVATION --dropout $DROPOUT_RATE --augment $AUGMENT --epochs $EPOCHS

    mkdir -p -m=776 Evaluation_logs/$MODEL_TYPE

    #Evaluate model and save results in eval/MODEL_NAME.txt file
    python3 evaluate_model.py --data_path $DATA_PATH --model_type $MODEL_TYPE --model_name $MODEL_NAME \
    --backbone $BACKBONE --loss $LOSS >> Evaluation_logs/$MODEL_TYPE/$MODEL_NAME.txt

    if [ $PREDICT = 'true' ]
    then
        # make predictions with the validation set and convert them to rgb
        python3 create_predictions.py --data_path $DATA_PATH --model_type $MODEL_TYPE --model_name $MODEL_NAME --backbone $BACKBONE --split "val"
        python3 convert2rgb.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --split "val"

        # make predictions with the test set and convert them to rgb
        python3 create_predictions.py --data_path $DATA_PATH --model_type $MODEL_TYPE --model_name $MODEL_NAME --backbone $BACKBONE --split "test"
        python3 convert2rgb.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --split "test"

        # zip the generated images and place the compressed file into the archives folder
        zip -r archives/$MODEL_TYPE-$MODEL_NAME.zip predictions/$MODEL Evaluation_logs/$MODEL.txt Confusion_matrix/$MODEL.png
    fi
}

if [ $HELP = 'true' ]
then
    displayHelp
else
    main
fi