echo "Machine Learning HomeWork A2 -- Convolutional Network for CIFAR10 Classification"
epochs=100
batch_size=64
lr=0.001
task="A2"
layer_norm_methods=(
    "layer_norm"
    "batch_norm"
)
max_seed=1
for ((i=1; i<=max_seed; i++)); do
    # baseline
    nohup python3 ./scripts/mains/a2_main.py --task $task --batch_size $batch_size \
        --epochs $epochs --eval_interval 3 -seed $i  >> "./logs/null_null_seed${i}.log" 2>&1 &    

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


