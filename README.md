# AGent: A Novel Pipeline for Automatically Creating Unanswerable Questions
## Introduction
This repository contains the source code for the AGent pipeline described in the following paper:
>**AGent: A Novel Pipeline for Automatically Creating Unanswerable Questions**<br>
>Son Quoc Tran, Gia-Huy Do, Phong Nguyen-Thuan Do, Matt Kretchmar, Xinya Du<br>
>Computer Science Department, Denison University, Granville, Ohio<br>
>The UIT NLP Group, Vietnam National University, Ho Chi Minh City<br>
>University of Texas at Dallas

<img src="pipeline.png" alt="Pipeline" width="1300"/>

## 1. Getting Started
```
conda create -n agent -y python=3.9.12
conda activate agent
pip install -r requirements.txt
```

## 2. Preparing Data
The original data should be in directory `src\step1\data\SQuAD\original`


## 3. AGent
<img src="example.png" alt="Example" width="800"/>
Agent pipeline has three steps:<br>
    1. Matching questions with new contexts.<br>
    2. Identifying hard unanswerable questions.<br>
    3. Filtering out answerable questions.

Note: this README is written as an example for creating SQuAD AGent. The SQuAD train and dev sets are saved at `src/step1/data/SQuAD/original/squad_train.json` and `src/step1/data/SQuAD/original/squad_dev.json`

### Step 1
Code for this step is in `src/step1`. Refer to `src/step1/README.md` for further instructions.


```
cd src/step1
command/phase1.sh
```

with the following arguments:

```
    --dataset_path ${DATASET_PATH} \
    --save_path ${SAVE_PATH} \
    --top_k 10 \
    --gt_score False \
    --num_unanswerable 300
```

The unanswerable candidates will be saved as:

```
src/step1/retriever_component/unans_cdd/{dataset_name}_data_cdd.json
```

The training file for adversarial models in step 2 is created by combining newly created unanswerable questions with the original answerable dataset and will be saved under `--save_path`.


Acknowledgement: Code for this step is mostly adopted from the repository of [DrQA](https://github.com/facebookresearch/DrQA) by [Chen et al., (2017)](https://aclanthology.org/P17-1171/).

### Step 2
#### Train Adversarial Models
The following command run in the original directory.
```
TRAIN_PATH="src/step1/data/SQuAD/adversarial/squad_train_step2.json"
SAVE_PATH="Model/Adversarial_Models"
for model in bert-base-cased bert-large-cased roberta-base roberta-large SpanBERT/spanbert-base-cased SpanBERT/spanbert-large-cased
do 
    python src/run_qa.py \
        --model_name_or_path ${model} \
        --train_file "${TRAIN_PATH}" \
        --do_train \
        --per_device_train_batch_size 8 \
        --learning_rate 2e-5 \
        --num_train_epochs 2 \
        --max_seq_length 384 \
        --max_answer_length 128 \
        --doc_stride 128 \
        --save_steps 999999 \
        --overwrite_output_dir \
        --version_2_with_negative \
        --output_dir "${SAVE_PATH}/${model}"
done
```
Then, get the predictions of adversarial models on unanswerable candidates created in Step 1.
```
MODEL_PATH="Model/Adversarial_Models"
EVAL_PATH="src/step1/retriever_component/unans_cdd/squad_train_data_cdd.json"
SAVE_PATH="Prediction/Adversarial_Models/SQuAD_Train"
for model in bert-base-cased bert-large-cased roberta-base roberta-large SpanBERT/spanbert-base-cased SpanBERT/spanbert-large-cased
do 
    python src/run_qa.py \
        --model_name_or_path "${MODEL_PATH}/${model}" \
        --validation_file "${EVAL_PATH}" \
        --do_eval \
        --per_device_eval_batch_size 8 \
        --max_seq_length 384 \
        --max_answer_length 128 \
        --doc_stride 128 \
        --n_best_size 5 \
        --overwrite_output_dir \
        --version_2_with_negative \
        --output_dir "${SAVE_PATH}/${model}"
done
```

#### Get Challenging Unanswerable Candidates
Use notebook `src/step2.ipynb` to finalize the Agent dataset.

Modify the following code cell to use AGent on other dataset.
```
tfidf_path = "step1/retriever_component/unans_cdd/squad_train_unans_cdd.json"
pred_path = "Prediction/Adversarial_Models"
save_path = "step1/data/SQuAD/challenging_unans_candidate/squad_train.json"
```
### Step 3
Use notebook `src/step3.ipynb` to tune the formula and finalize the Agent dataset.

Modify the following code cell to use AGent on other dataset.
```
annotated_pred_path = "Prediction/Step3_Annotated"
answerable_ids_path = "answerable_ids.json"

full_pred_dev_path = "src/step1/retriever_component/unans_cdd/squad_dev_unans_cdd.json"      # path to predictions of the 6 adversarial models
dev_path = "Prediction/Adversarial_Models/SQuAD_Dev"    # path to all challenging unanswerable candidates (product of step 2)
squadv1_dev_path = "src/step1/data/SQuAD/original/squad_dev.json"
save_dev_path = "AGent_data/SQuAD/dev.json"


tfidf_train_path = "src/step1/retriever_component/unans_cdd/squad_train_unans_cdd.json"
full_pred_train_path = "Prediction/Adversarial_Models/SQuAD_Train"
squadv1_train_path = "src/step1/data/SQuAD/original/squad_train.json"
save_train_path = "AGent_data/SQuAD/train.json"
```
## Citation and Contact
If you found this repository helpful, please cite:
```
@misc{tran2023agent,
      title={AGent: A Novel Pipeline for Automatically Creating Unanswerable Questions}, 
      author={Son Quoc Tran and Gia-Huy Do and Phong Nguyen-Thuan Do and Matt Kretchmar and Xinya Du},
      year={2023},
      eprint={2309.05103},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
Please contact Son Quoc Tran at `tran_s2@denison.edu` if you have any questions.

