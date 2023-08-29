# Step 1:

Use command/phase1.sh file to convert the answerable dataset int unanswerable candidate dataset (dataset_name_unans_cdd.json)

```
python /Volumes/Share/tran_s2/AGent_Hub/All_code_submission/unans_candidate/TF-IDF/main.py \
    --dataset_path ${DATASET_PATH} \
    --save_path ${SAVE_PATH} \
    --top_k 10 \
    --gt_score False \
    --num_unanswerable 300
```


--dataset_path: Path to the answerable dataset <br>
--save_path: Path to save the training file for phase 2 <br>
--top_k: Number of context from TF-IDF <br>
--gt_score (default = False): Parameter to print the score of the TF-IDF of answerable context - question <br>
--num_unanswerable (default = False): select random number of unanswerable candidates to combine with the answerable dataset <br>

The flow of the dataset in phase 1 is presented as follows: <br>
1. The dataset is parsed through the main.py file, then the information of the answerable dataset is retrieved <br>
2. The TF-IDF ranker is deployed:
    1. If the --gt_score=True, the ground truth score of the answerable dataset is saved under unans_candidate/TF-IDF/retriever_component/tfidf_data/gt_score
    2. If the --gt_score=False, we prepare the db form and database of the answerable dataset, then save it under unans_candidate/TF-IDF/retriever_component/db_form and unans_candidate/TF-IDF/retriever_component/SQLdatabase/NQ_mrqa.db <br>
    3. Save top_k contexts under unans_candidate/TF-IDF/retriever_component/tfidf_data/relevant
3. The top_k context-question under unans_candidate/TF-IDF/retriever_component/tfidf_data/relevant is parsed through the unans_cdd.py file for re-formatting purposes. The final database of all unans_candidates is saved at unans_candidate/TF-IDF/retriever_component/unans_cdd/.
4. The combine.py file is used to combine the answerable datasets with --num_unanswerable of the unanswerable candidates dataset to prepare for phase 2.
# Step 2
1. Train the models
2. Get predictions
3. Get adversarial unanswerable candidates
# Step 3
Tune the model and select qualified samples