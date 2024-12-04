echo "Machine Learning Homework A3"
seeds=(1)
work_dir="/home/weijun.luo/ml_homework"
cd work_dir
name_dir="/home/weijun.luo/ml_homework/data/names"
for seed in "${dirs[@]}"; do
    LOG_FILE="./logs/rnn_${seed}.log"
    CUDA_VISIBLE_DEVICES=0,2,3 nohup python scripts/mains/a3_main.py --task A3 --name_dir $name_dir --seed $seed >> $LOG_FILE 2>&1 &
done