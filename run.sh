#! /bin/bash
set -e

numArg=$#
HELP=false

# set Default values
BACKBONE=None
TRAIN_OUTPUT_STRIDE=32
EVAL_OUTPUT_STRIDE=32
LOSS=FocalHybridLoss
BATCH_SIZE=3
ACTIVATION=leaky_relu
DROPOUT_RATE=0.0
AUGMENT=false
EPOCHS=20
FINAL_EPOCHS=60
TRAIN=true
EVAL=true
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
    
    if [ "$1" == "--train-out-stride" ]
        then
            TRAIN_OUTPUT_STRIDE=$2
        fi

    if [ "$1" == "--eval-out-stride" ]
        then
            EVAL_OUTPUT_STRIDE=$2
        fi

    if [ "$1" == "-l" ] || [ "$1" == "--loss" ]
        then
            LOSS=$2
        fi

    if [ "$1" == "--batch-size" ]
        then
            BATCH_SIZE=$2
        fi

    if [ "$1" == "--activation" ]
        then
            ACTIVATION=$2
        fi

    if [ "$1" == "--dropout" ]
        then
            DROPOUT_RATE=$2
        fi

    if [ "$1" == "--augment" ]
        then
            AUGMENT=true
        fi

    if [ "$1" == "-e" ] || [ "$1" == "--epochs" ]
        then
            EPOCHS=$2
        fi

    if [ "$1" == "--final-epochs" ]
        then
            FINAL_EPOCHS=$2
        fi

    if [ "$1" == "--no-train" ] || [ "$1" == "--no-training" ]
        then
            TRAIN=false
        fi

    if [ "$1" == "--no-eval" ] || [ "$1" == "--no-evaluation" ]
        then
            EVAL=false
        fi

    if [ "$1" == "-p" ] || [ "$1" == "--predict" ]
        then
            PREDICT=true
        fi

    #shift 'consumes' the argument and then the loop can move to the next one
    shift
done

MODEL=$MODEL_TYPE/$MODEL_NAME

displayHelp(){
    echo ''
    echo 'Usage: ./run.sh [OPTIONS]'
    echo ''
    echo 'Perform model training, evaluation and optionally create the predictions'
    echo ''
    echo 'Options:'
    echo '  -h, --help               Display help'
    echo '  -d, --data-path          The root directory of the dataset'
    echo '  -t, --model-type         Model type'
    echo '  -n, --model-name         The name the model will have'
    echo '  -b, --backbone           The backbone that will be used for the model. Supported Backbones are ResNet, ResNetV2, EfficientNet, EfficientNetV2, MobileNetV1,V2,V3 and RegNetX and RegNetY. (Defaults to None).'
    echo '      --train-out-stride   The output stride to use during training. Output stride is the ratio of input image spatial resolution to the encoder output resolution. (Defaults to 32).'
    echo '      --eval-out-stride    The output stride to use during inference/evaluation.  (Defaults to 32).'
    echo '  -l, --loss               Loss function to be used for training the model. (Options: DiceLoss, IoULoss, TverskyLoss, FocalTverskyLoss, HybridLoss, FocalHybridLoss) (Defaults to FocalHybridLoss).'
    echo '      --batch-size         Batch size to be used during training.  (Defaults to 3).'
    echo '      --activation         Activation function to be used at the output of each Conv2D layer. (Defaults to leaky_relu).'
    echo '      --dropout            Dropout rate of the dropout layers. (Defaults to 0).'
    echo '      --augment            Use data augmentation.  (Defaults to false).'
    echo '  -e, --epochs             The number of epochs the model will be trained for. When using a model backbone this number is the number of epochs for the initial run where the backbone is frozen. Defaults to 20'
    echo '      --final-epochs       The final number of epochs for the second run where part of the backbone is unfrozen. Defaults to 60'
    echo '      --no-train           Set this flag to disable training.'
    echo '      --no-eval            Set this flag to disable evaluation.'
    echo '  -p, --predict            Whether to make predictions or not for val and test sets after training and evaluating the model.'
}

main(){
    # train model
    if [ $TRAIN = 'true' ]
    then
        cd scripts/ && python3 train_model.py --data_path $DATA_PATH --model_type $MODEL_TYPE --model_name $MODEL_NAME --backbone $BACKBONE --output_stride $TRAIN_OUTPUT_STRIDE\
        --loss $LOSS --batch_size $BATCH_SIZE --activation $ACTIVATION --dropout $DROPOUT_RATE --augment $AUGMENT --epochs $EPOCHS --final_epochs $FINAL_EPOCHS
    fi

    #Evaluate model
    if [ $EVAL = 'true' ]
    then
        mkdir -p -m=776 Evaluation_logs/$MODEL_TYPE

        cd scripts/ && python3 evaluate_model.py --data_path $DATA_PATH --model_type $MODEL_TYPE --model_name $MODEL_NAME --output_stride $EVAL_OUTPUT_STRIDE\
        --backbone $BACKBONE --loss $LOSS >> Evaluation_logs/$MODEL_TYPE/$MODEL_NAME.txt
    fi

    if [ $PREDICT = 'true' ]
    then
        # make predictions with the validation set and convert them to rgb
        cd scripts/ && python3 create_predictions.py --data_path $DATA_PATH --model_type $MODEL_TYPE --model_name $MODEL_NAME --backbone $BACKBONE --split "val"

        # make predictions with the test set and convert them to rgb
        cd scripts/ && python3 create_predictions.py --data_path $DATA_PATH --model_type $MODEL_TYPE --model_name $MODEL_NAME --backbone $BACKBONE --split "test"

        # zip the generated images and place the compressed file into the archives folder
        zip -r archives/$MODEL_TYPE-$MODEL_NAME.zip predictions/$MODEL Evaluation_logs/$MODEL.txt saved_models/$MODEL
    fi
}

if [ $HELP = 'true' ]
then
    displayHelp
else
    main
fi