OUTPUT=outputs
SHOTS=16

SOURCE=busi
TARGETS=(buid busbra udiat)

python main.py --root_path data --dataset ${SOURCE} --tasks 3 \
                --shots ${SHOTS} \
                --output_dir ${OUTPUT} \
                --save_path ${OUTPUT} \
                --config configs/EviSteer.yaml

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
TARGETS=(btmri_p btmri_s brisc)

python main.py --root_path data --dataset ${SOURCE} --tasks 3 \
                --shots ${SHOTS} \
                --output_dir ${OUTPUT} \
                --save_path ${OUTPUT} \
                --config configs/EviSteer.yaml

for DATASET in "${TARGETS[@]}"
do
python main.py --root_path data --dataset ${DATASET} --tasks 3 \
                --shots ${SHOTS}  \
                --output_dir ${OUTPUT} \
                --save_path ${OUTPUT}/${SOURCE} \
                --config configs/EviSteer.yaml \
                --eval_only  
done

