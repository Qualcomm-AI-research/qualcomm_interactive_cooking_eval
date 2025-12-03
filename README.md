# Qualcomm Interactive Cooking Dataset Evaluator

<p align="center">
  <img width="640" src="/assets/dataset.png" hspace="50">
</p>

Here we provide the code to evaluate models on the Qualcomm Interactive Cooking Dataset in the paper:

[**Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance? (NeurIPS 2025)**](https://openreview.net/pdf?id=G6K2NepP7S)

## Abstract

Multi-modal Large Language Models (LLM) have advanced conversational abilities but struggle with providing live, interactive step-by-step guidance, a key capability for future AI assistants. Effective guidance requires not only delivering instructions but also detecting their successful execution, as well as identifying and alerting users to mistakes, all of which has to happen in real-time. This requires models that are not turn-based, but that can react asynchronously to a video stream, as well as video data showing users performing tasks including mistakes and their corrections. To this end, we introduce Qualcomm Interactive Cooking, a new benchmark and dataset built upon CaptainCook4D, which contains user mistakes during task execution. Our dataset and benchmark features densely annotated, timed instructions and feedback messages, specifically including mistake alerts precisely timestamped to their visual occurrence in the video. We evaluate state-of-the-art multi-modal LLMs on the Qualcomm Interactive Cooking benchmark and introduce LiveMamba, a streaming multi-modal LLM designed for interactive instructional guidance. This work provides the first dedicated benchmark and a strong baseline for developing and evaluating on live, situated coaching.

## Running the Code

### Getting Started

First, clone the repository using the following command:

```
git clone https://github.com/Qualcomm-AI-research/QualcommInteractiveCookingEvaluator.git $REPO_PATH
cd $REPO_PATH
```

Here, *$REPO_PATH* is the desired location to download the repository.

Next, build a conda environment with the project requirements using the following commands, where `<env>` is your desired environment name:

```
conda create --name <env> python=3.11.10
conda activate <env>
conda install bert_score rouge-score tqdm -c conda-forge
pip install huggingface-hub==0.34.3 datasets==3.3.2
```

### Evaluation

For evaluation, create a json file with the predictions, which should be in the following format,
```
[
    {
        "video_id": <ID of the video in the Qualcomm Interactive Cooking Dataset>,
        "pred_texts": <List of predicted instructions and feedbacks with index 0 being the instruction>,
        "pred_timestamps": <Timestamps corresponding to the instructions and feedbacks above>
    },
    ...
]
```

Note that, every string in `pred_texts` should have one of the following three prefixes: Instruction, Feedback or Success -- "Instruction: <>", "Feedback: <>" or "Success: <>".

Next use the following command:

```
PYTHONPATH=./ python eval.py \
    --plan_set <"main" or "advanced_planning"> \
    --split <"train","validation" or "test"> \
    --predictions_file_path <path to the json file with the predictions described above>
```

Note: This code downloads the [Qualcomm Interactive Cooking Dataset](https://huggingface.co/datasets/qualcomm/qualcomm-interactive-cooking-dataset) automatically. The code was tested on a single Nvidia A100 GPU.

## Repository Structure

The repository has the following structure,

```text
qualcomm_live_cooking_eval
|   assets/ : Images in README
|   └── dataset.png
└───data.py : Loads the Qualcomm Interactive Cooking Dataset
└───eval.py : Runs the evaluation and calculates the IC-Acc and Mistake Detection Metrics
└───utils.py : misc functions
```

## License

The code in this repository is released under the BSD-3 Clause Clear license. Please refer to LICENSE for details.

## Citation

```text
@inproceedings{interactivecooking,
   title = {Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?},
   author = {Apratim Bhattacharyya and Bicheng Xu and Sanjay Haresh and Reza Pourreza and Litian Liu and Sunny Panchal and Leonid Sigal and Roland Memisevic},
   booktitle = {NeurIPS},
   year = {2025},
}
```