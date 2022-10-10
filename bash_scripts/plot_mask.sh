cd ../code/visualize;

# note that the config below should match your checkpoint's config, such as prune-rate, pr-scale, etc.
python weight_mask_new.py \
    --file /path/to/checkpoint \
    --config /path/to/the/same/config/file \
    --model_name checkpoint_name \
    --multigpu 0 \
    --prune-rate 0.01 \
    --pr-scale 0.1 \
    --end-with-bn \
    --score-init-scale 0.001 \
    --fan-scaled-score-mode none;
