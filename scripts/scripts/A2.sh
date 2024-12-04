echo "Machine Learning HomeWork A2 -- Convolutional Network for CIFAR10 Classification"
epochs=100
batch_size=64
lr=0.001
task="A2"
pool_methods=(
    "avg_pool"
    "max_pool"
)
layer_norm_methods=(
    "layer_norm"
    "batch_norm"
)


# relevant dir
log_dir="./logs"
res_dir="./result"
data_dir="./data"
runs_dir="./runs"
dirs=(
    "$log_dir"
    "$res_dir"
    "$data_dir"
    "$runs_dir"
)
# if pwd isn't exist, create it.
for dir in "${dirs[@]}"; do
    if [ ! -d $dir ]; then
        mkdir -p $dir
	echo "${dir} will be created!"
    else
        echo "${dir} has existed. There is no need for recreate it!"
    fi
done

# start training process
max_seed=1
for ((i=1; i<=max_seed; i++)); do
    # baseline
    nohup python3 ./scripts/mains/a2_main.py --task $task --batch_size $batch_size \
        --epochs $epochs --eval_interval 3 -seed $i  >> "./logs/null_null_seed${i}.log" 2>&1 &
    
    # compare different pool method
    nohup python3 ./scripts/mains/a2_main.py --task $task --batch_size $batch_size \
        --epochs $epochs --eval_interval 3 -seed $i --pool_method  "max_pool" >> "./logs/null_null_seed${i}.log" 2>&1 &

    nohup python3 ./scripts/mains/a2_main.py --task $task --batch_size $batch_size \
        --epochs $epochs --eval_interval 3 -seed $i --pool_method  "avg_pool" >> "./logs/null_null_seed${i}.log" 2>&1 &

    # compare different normalization, three types (dropout, batchnorm, layernorm)
    nohup python3 ./scripts/mains/a2_main.py --task $task --batch_size $batch_size \
            --epochs $epochs --eval_interval 3 -seed $i --use_layer_norm --layer_norm_method "layer_norm"  >> "./logs/layernorm_null_seed${i}.log" 2>&1 &


    nohup python3 ./scripts/mains/a2_main.py --task $task --batch_size $batch_size \
            --epochs $epochs --eval_interval 3 -seed $i --use_layer_norm >> "./logs/batchnorm_null_seed${i}.log" 2>&1 &

    nohup python3 ./scripts/mains/a2_main.py --task $task --batch_size $batch_size \
        --epochs $epochs --eval_interval 3 -seed $i --use_dropout  >> "./logs/dropout_null_seed${i}.log" 2>&1 &

    # compare diffent regularizations, two type (L1-regular, L2-regular)
    nohup python3 ./scripts/mains/a2_main.py --task $task --batch_size $batch_size \
        --epochs $epochs --eval_interval 3 -seed $i --use_l1_norm  >> "./logs/null_l2_seed${i}.log" 2>&1 &

    nohup python3 ./scripts/mains/a2_main.py --task $task --batch_size $batch_size \
        --epochs $epochs --eval_interval 3 -seed $i --use_l2_norm  >> "./logs/null_l2_seed${i}.log" 2>&1 &
done


