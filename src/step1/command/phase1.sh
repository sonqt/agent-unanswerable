export PYTHONPATH="${PYTHONPATH}:agent-unanswerable/src/step1/unans_candidate/TF-IDF/DrQA"
DATASET_PATH="agent-unanswerable/src/step1/Data/NQ/NQ_mrqa_dev.json"
SAVE_PATH="agent-unanswerable/src/step1/Data/NQ/NQ_mrqa_dev_unans_cdd.json"

python agent-unanswerable/src/step1/unans_candidate/TF-IDF/main.py \
    --dataset_path ${DATASET_PATH} \
    --save_path ${SAVE_PATH} \
    --top_k 5 \
    --gt_score False \
    --num_unanswerable 300
