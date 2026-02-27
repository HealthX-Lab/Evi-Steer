OUTPUT=checkpoints
SHOTS=16

SOURCE=busi
TARGETS=(busi buid busbra busb udiat)

for DATASET in "${TARGETS[@]}"
do
python main.py --root_path data --dataset ${DATASET} --tasks 3 \
                --shots ${SHOTS}  \
                --output_dir ${OUTPUT} \
                --save_path ${OUTPUT}/${SOURCE} \
                --config configs/EviSteer.yaml \
                --eval_only  
done

SOURCE=btmri
TARGETS=(btmri btmri_p btmri_s brisc)

for DATASET in "${TARGETS[@]}"
do
python main.py --root_path data --dataset ${DATASET} --tasks 3 \
                --shots ${SHOTS}  \
                --output_dir ${OUTPUT} \
                --save_path ${OUTPUT}/${SOURCE} \
                --config configs/EviSteer.yaml \
                --eval_only  
done