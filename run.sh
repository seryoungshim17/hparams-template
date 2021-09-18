count=0
for LR in 0.01 0.001 0.0001;
do
    for OPT in Adam SGD;
    do
    count=$((${count}+1))
    python main.py \
    --EPOCH 1 \
    --OPTIMIZER ${OPT} \
    --LR ${LR} \
    --EXP_NUM ${count} \
    --BATCH_SIZE 64
    done
done