# AGent: A Novel Pipeline for Automatically Creating Unanswerable Questions
## Introduction
This repository contains the source code for the AGent pipeline described in the following paper:
>**AGent: A Novel Pipeline for Automatically Creating Unanswerable Questions**<br>
>Son Quoc Tran, Gia-Huy Do, Phong Nguyen-Thuan Do, Matt Kretchmar, Xinya Du<br>
>Computer Science Department, Denison University, Granville, Ohio<br>
>The UIT NLP Group, Vietnam National University, Ho Chi Minh City<br>
>University of Texas at Dallas

![The AGent pipeline for generating challenging high-quality unanswerable questions in Extractive Question Answering given a dataset with answerable questions. The six models used in this pipeline are the base and large versions of BERT, RoBERTa, and SpanBERT. In step 3 of the pipeline, the blue dots represent the calculated values (using formula in step 3) for unanswerable questions, while the red dots represent the calculated values for answerable questions. The threshold for discarding questions from the final extracted set of unanswerable questions is determined by finding the minimum value among all answerable questions. Any question with a calculated value greater than the threshold will not be included in our final extracted set.](pipeline.png)
## 1. Getting Started
```
conda create -n agent -y python=3.9.12
conda activate agent
pip install -r requirements.txt
```

## 2. Preparing Data

## 3. AGent
![Examples of an answerable question $Q1$ from SQuAD 1.1, and two unanswerable questions $Q2$ from SQuAD 2.0 and $Q3$ from SQuAD *AGent*. In SQuAD 2.0,  crowdworkers create unanswerable questions by replacing ``large numbers'' with ``decimal digits.'' On the other hand, our automated *AGent* pipeline matches the original question $Q1$, now $Q3$, with a new context $C3$. The pair $C3-Q3$ is unanswerable as context $C3$ does not indicate whether the **trial division** can **conveniently** test the primality of **large** numbers.](example.png)
Agent pipeline has three steps:<br>
    1. Matching questions with new contexts.<br>
    2. Identifying hard unanswerable questions.<br>
    3. Filtering out answerable questions.<br>
### Step 1
This step is mostly adopted from the repository of [DrQA](https://github.com/facebookresearch/DrQA) by [Danqi Chen et al., 2017](https://aclanthology.org/P17-1171/).

Code for this step is in `src/step1`. Refer to `src/step1/README.md` for further instructions.
### Step 2
#### Train Adversarial Models
#### Get Challenging Unanswerable Candidates
Use notebook `src/step2.ipynb` to finalize the Agent dataset
### Step 3
Use notebook `src/step3.ipynb` to tune the formula and finalize the Agent dataset

## Citation and Contact
If you found this repository helpful, please cite:
```
BibText here
```
Please contact Son Quoc Tran at `tran_s2@denison.edu` if you have any questions.

