gpu=0,1

pr=0.01
d=0.1
checkpoint_dir=/path/to/store/pretrained/checkpoints

cd ../code;

i=0;

for dataset in cifar10 cifar100; do
    for layer in 18 34 50; do
        model=resnet$layer
        if [ "$dataset" = cifar100 ]; then
            model_type=cResNet${layer}_cifar100
        else
            model_type=cResNet${layer}
        fi
        for fast_train in y n; do
            ckpt=rn${layer}_${dataset}_${fast_train}

            config_file=configs/advprune/$dataset/$model/aa-standard.yaml

            pretrained=$checkpoint_dir/$ckpt.pth

            python run/aa_standard.py \
                --config $config_file \
                --model_name camera-$ckpt \
                --model_type $model_type \
                --multigpu $gpu \
                --pretrained $pretrained \
                --prune-rate $pr \
                --pr-scale $d \
                --end-with-bn;
        done
    done
done

for dataset in imagenet100; do
    for layer in 34; do
        model=resnet$layer
        model_type=ResNet34_imagenet100

        ckpt=rn${layer}_${dataset}

        config_file=configs/advprune/$dataset/$model/aa-standard.yaml

        pretrained=$checkpoint_dir/$ckpt.pth

        python run/aa_standard.py \
            --config $config_file \
            --model_name camera-$ckpt \
            --model_type $model_type \
            --multigpu $gpu \
            --pretrained $pretrained \
            --prune-rate $pr \
            --pr-scale $d \
            --end-with-bn;
    done
done