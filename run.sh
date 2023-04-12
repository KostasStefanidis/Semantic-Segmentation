#! /bin/bash
set -e

numArg=$#
HELP=false

# set Default values
DATASET=Cityscapes
TRAIN_OUTPUT_STRIDE=32
EVAL_OUTPUT_STRIDE=32
OPTIMIZER=Adam
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
    if [ "$1" == "--config" ]
        then
            CONFIG=$2
        fi

    if [ "$1" == "-h" ] || [ "$1" == "--help" ]
        then
            HELP=true
        fi

    if [ "$1" == "--dataset" ]
        then
            DATASET=$2
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
            BACKBONE=$2 #Mandatory
        fi
    
    if [ "$1" == "--unfreeze-at" ]
        then
            UNFREEZE_AT=$2 #mandatory
        fi

    if [ "$1" == "--train-out-stride" ]
        then
            TRAIN_OUTPUT_STRIDE=$2
        fi

    if [ "$1" == "--eval-out-stride" ]
        then
            EVAL_OUTPUT_STRIDE=$2
        fi

    if [ "$1" == "-o" ] || [ "$1" == "--optimizer" ]
        then
            OPTIMIZER=$2
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
    echo '      --dataset            Which dataset to use. (Cityscapes, Mapillary) (default: Cityscapes)'
    echo '  -d, --data-path          The root directory of the dataset'
    echo '  -t, --model-type         Model type'
    echo '  -n, --model-name         The name the model will have'
    echo '  -b, --backbone           The backbone that will be used for the model. (Options: ResNet, ResNetV2, EfficientNet, EfficientNetV2, MobileNetV1,V2,V3 and RegNet).'
    echo '      --unfreeze-at        Where to unfreeze the network. Essentially the name of the layer up to which the error will be propagated buring backpropagation.'
    echo '      --train-out-stride   The output stride to use during training. Output stride is the ratio of input image spatial resolution to the encoder output resolution. (default 32).'
    echo '      --eval-out-stride    The output stride to use during inference/evaluation.  (default 32).'
    echo '  -o  --optimizer          The optimization algorithm used to train the network (default Adam)'
    echo '  -l, --loss               Loss function to be used for training the model. (Options: DiceLoss, IoULoss, TverskyLoss, FocalTverskyLoss, HybridLoss, FocalHybridLoss) (default FocalHybridLoss).'
    echo '      --batch-size         Batch size to be used during training. (default 3).'
    echo '      --activation         Activation function to be used at the output of each Conv2D layer. (default leaky_relu).'
    echo '      --dropout            Dropout rate of the dropout layers. (default 0).'
    echo '      --augment            Use data augmentation.  (default false).'
    echo '  -e, --epochs             The number of epochs the model will be trained for. When using a model backbone this number is the number of epochs for the initial run where the backbone is frozen. (default 20)'
    echo '      --final-epochs       The final number of epochs for the second run where part of the backbone is unfrozen. (default 60)'
    echo '      --no-train           Set this flag to disable training.'
    echo '      --no-eval            Set this flag to disable evaluation.'
    echo '  -p, --predict            Whether to make predictions or not for val and test sets after training and evaluating the model.'
}


main_with_config(){
    # train model
    if [ $TRAIN = 'true' ]
    then
        python3 train.py --config $CONFIG
    fi

    #Evaluate model
    if [ $EVAL = 'true' ]
    then
        mkdir -p -m=776 Evaluation_logs/$MODEL_TYPE
        python3 evaluate.py --config $CONFIG
    fi

    if [ $PREDICT = 'true' ]
    then
        # make predictions with the validation set and convert them to rgb
        python3 predict.py --config $CONFIG --split "val"

        # make predictions with the test set and convert them to rgb
        python3 predict.py --config $CONFIG --split "test"

        # zip the generated images and place the compressed file into the archives folder
        zip -r archives/$MODEL_TYPE-$MODEL_NAME.zip predictions/$MODEL Evaluation_logs/$MODEL.txt saved_models/$MODEL
    fi
}

main_with_args(){
    # train model
    if [ $TRAIN = 'true' ]
    then
        python3 train.py --data_path $DATA_PATH --model_type $MODEL_TYPE --model_name $MODEL_NAME --backbone $BACKBONE --output_stride $TRAIN_OUTPUT_STRIDE\
        --loss $LOSS --batch_size $BATCH_SIZE --activation $ACTIVATION --dropout $DROPOUT_RATE --augment $AUGMENT --epochs $EPOCHS --final_epochs $FINAL_EPOCHS\
        --optimizer $OPTIMIZER --unfreeze_at $UNFREEZE_AT --dataset $DATASET
    fi

    #Evaluate model
    if [ $EVAL = 'true' ]
    then
        mkdir -p -m=776 Evaluation_logs/$MODEL_TYPE

        python3 evaluate.py --data_path $DATA_PATH --model_type $MODEL_TYPE --model_name $MODEL_NAME --output_stride $EVAL_OUTPUT_STRIDE\
        --backbone $BACKBONE --loss $LOSS >> Evaluation_logs/$MODEL_TYPE/$MODEL_NAME.txt
    fi

    if [ $PREDICT = 'true' ]
    then
        # make predictions with the validation set and convert them to rgb
        python3 predict.py --data_path $DATA_PATH --model_type $MODEL_TYPE --model_name $MODEL_NAME --backbone $BACKBONE --split "val"

        # make predictions with the test set and convert them to rgb
        #python3 predict.py --data_path $DATA_PATH --model_type $MODEL_TYPE --model_name $MODEL_NAME --backbone $BACKBONE --split "test"

        # zip the generated images and place the compressed file into the archives folder
        #zip -r archives/$MODEL_TYPE-$MODEL_NAME.zip predictions/$MODEL Evaluation_logs/$MODEL.txt saved_models/$MODEL
    fi
}

if [ $HELP = 'true' ]
then
    displayHelp
elif [ -z "$CONFIG" ]
then
    if [ -z "$DATA_PATH" ]
    then
        echo 'No Data path defined. Use the -d or --data-path option.'
        exit 125
    fi

    if [ -z "$MODEL_TYPE" ]
    then
        echo 'No Model Type defined. Use the -t or --model-type option.'
        exit 125
    fi

    if [ -z "$MODEL_NAME" ]
    then
        echo 'No Model Name defined. Use the -n or --model-name option.'
        exit 125
    fi

    if [ -z "$BACKBONE" ]
    then
        echo 'No Backbone defined. Use the -b or --backbone option.'
        exit 125
    fi

    if [ -z "$UNFREEZE_AT" ]
    then
        echo 'No unfreze_at defined. Use the --unfreeze-at option.'
        exit 125
    fi

    main_with_args
else
    main_with_config
fi