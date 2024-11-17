echo "Machine Learning Homework A1"
optimizer=(
    # "adam"
    # "sgd"
    # "adamw"
    # "adagrad"
    # "adamax"
    "rmsprop"
    # "momentum"
)
    
loss=(
    "entropy"
    # "mse"
)
task="A1"
epochs=100
batch_size=64
max_seed=1

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
    else
        echo "${dir} has existed. There is no need for recreate it!"
    fi
done



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
