OUTPUT=checkpoints

for DATASET in busi btmri chmnist covid ctkidney kvasir retina octmnist lungcolon chmnist 
do
for SHOTS in 4 8 16
do
python main.py --root_path data --dataset ${DATASET} --tasks 3 --shots ${SHOTS} \
                --output_dir ${OUTPUT} --config configs/EviSteer.yaml --save_path ${OUTPUT}/${DATASET} --eval_only
done
done