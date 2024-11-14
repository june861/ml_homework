echo "Machine Learning Homework A1"
optimizer=(
    # "adam"
    "sgd"
    # "adamw"
    # "adagrad"
    # "momentum"
)
    
loss=(
    "entropy"
    # "mse"
)
task="A1"
epochs=200
batch_size=64
max_seed=3
log_dir="./logs"
# if pwd isn't exist, create it.
mkdir -p "$log_dir"

for optim in "${optimizer[@]}"; do
    for lo in "${loss[@]}"; do
        for ((i=1; i<=max_seed; i++)); do
            LOG_FILE="${log_dir}/${optim}_${lo}_${i}.log"
            echo "optimizer is $optim, loss is $lo, batch size is $batch_size"
            echo "log will be output to $LOG_FILE"
            nohup python3 ./scripts/mains/a1_main.py --task $task --loss $lo --optim $optim --batch_size $batch_size \
                    --epochs $epochs --eval_interval 5 -seed $i >> $LOG_FILE 2>&1 &
        done
    done
done
