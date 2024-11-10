echo "Machine Learning Homework A1"
optimizer=(
    "adam",
    "sgd",
    "adamw",
    "adagrad",
    "momentum",
)

loss=(
    "entropy",
    "mse",
)
epochs=50
batch_size=64
max_seed=1
log_dir="./logs"
# if pwd isn't exist, create it.
mkdir -p "$log_dir"

for optim in "${optimizer[@]}"; do
    for lo in "${loss[@]}"; do
        for ((i=1; i<=max_seed; i++)); do
            LOG_FILE="${log_dir}/${optim}_${lo}_${seed}.log"
            echo "optimizer is $optim, loss is $lo, batch size is $batch_size"
            echo "log will be output to $LOG_FILE"
            nohup python3 ./A1/a1.py --loss $lo --optim $optim --batch_size $batch_size \
                    --epochs $epochs --eval_interval 5 -seed $i >> $LOG_FILE 2>&1 &
        done
    done
done
