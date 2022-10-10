dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt

date=$(date +%m%d)

train(){
    # order: model_name($1), gpu($2), epochs($3)
    cd ../code/;
    
    pr=001
    d=01

    model=resnet34
    dataset=cifar100
    model_type=ResNet34_${pr}_p${d}_cifar100
    model_name=$1-baseline-rn34-smallnet-bnn-at-bn-$dataset
    savedir=/path/to/store/the/experiments

    python run/train.py \
        --config configs/advprune/$dataset/conv/$model-$pr-p$d-bnn.yaml \
        --log-dir $savedir/ \
        --model_name $model_name \
        --multigpu $2 \
        --lr 0.0001 \
        --end-with-bn \
        --epochs $3;

    checkpoint_dir=$savedir/$model-$pr-p$d-bnn

    config_file=configs/advprune/$dataset/conv/aa-standard-binary.yaml

    pretrained=$checkpoint_dir/$model_name/prune_rate=1.0/0/checkpoints/model_best.pth

    python run/aa_standard.py \
        --config $config_file \
        --model_name $model_name \
        --model_type $model_type \
        --multigpu $2 \
        --pretrained $pretrained \
        --prune-rate 1.0 \
        --pr-scale 1.0 \
        --end-with-bn;

    cd ../bash_scripts/;
}

model_name=$date
gpu=2
epochs=400

(
    train $model_name $gpu $epochs;
);