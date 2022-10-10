dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt

date=$(date +%m%d)

train(){
    # order: model_name($1), gpu($2), epochs($3)
    cd ../code/;
    for d in 0.1; do
        for pr in 0.01; do
            model=resnet34
            dataset=cifar10
            model_type=cResNet34
            model_name=$1-rn34-ours-bn-$dataset
            savedir=/path/to/store/the/experiments

            python run/train.py \
                --config configs/advprune/$dataset/$model/$model-binary-weights.yaml \
                --log-dir $savedir/ \
                --model_name $model_name \
                --multigpu $2 \
                --prune-rate $pr \
                --pr-scale $d \
                --epochs $3 \
                --score-init-scale 0.01 \
                --fan-scaled-score-mode none \
                --end-with-bn \
                --debug;

            checkpoint_dir=$savedir/$model-binary-weights

            config_file=configs/advprune/$dataset/$model/aa-standard.yaml

            pretrained=$checkpoint_dir/$model_name/prune_rate=$pr/0/checkpoints/model_best.pth

            python run/aa_standard.py \
                --config $config_file \
                --model_name $model_name \
                --model_type $model_type \
                --multigpu $2 \
                --pretrained $pretrained \
                --prune-rate $pr \
                --pr-scale $d \
                --end-with-bn;
        done
    done
    cd ../bash_scripts/;
}

model_name=$date
gpu=0,1
epochs=400

(
    train $model_name $gpu $epochs;
);