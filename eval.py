# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import argparse
import json
from typing import Any, Dict, List, Tuple

from torch.utils.data import Dataset
from tqdm import tqdm

from utils import *

DETECTION_WINDOW_LENGTH = 15.0  # corresponding to a 30sec window
IGNORE_STRINGS = ["You are not following the instruction.", "You did not follow the instruction."]


def get_parser() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    _parser = argparse.ArgumentParser(
        description="",
        usage="python eval.py [--options]",
    )

    _parser.add_argument(
        "--plan_set",
        default="main",
        required=False,
        help="The eval set should be in ['main','advanced_planning'].",
    )

    _parser.add_argument(
        "--split",
        default="test",
        required=False,
        help="The eval split should be in ['train','validation','test'].",
    )

    _parser.add_argument(
        "--predictions_file_path",
        default=None,
        required=True,
        help="The path to file with the predictions.",
    )

    return _parser


def calculate_accuracy(tp: int, fp: int, tn: int, fn: int) -> float:
    """
    Calculates the overall accuracy.
    (TP + TN) / (TP + FP + TN + FN)
    Proportion of total predictions that were correct.
    """
    total = tp + fp + tn + fn
    if total == 0:
        return 0.0  # Avoid division by zero if no data
    return (tp + tn) / total


def calculate_precision(tp: int, fp: int, tn: int, fn: int) -> float:
    """
    Calculates the precision (Positive Predictive Value, PPV).
    TP / (TP + FP)
    Of those predicted as mistakes, how many actually were mistakes?
    High precision means fewer false alarms (FP).
    """
    predicted_positives = tp + fp
    if predicted_positives == 0:
        return 0.0  # Avoid division by zero if no positive predictions were made
    return tp / predicted_positives


def calculate_recall(tp: int, fp: int, tn: int, fn: int) -> float:
    """
    Calculates the recall (Sensitivity, True Positive Rate, TPR).
    TP / (TP + FN)
    Of all the actual mistakes, how many were correctly detected?
    High recall means fewer missed mistakes (FN).
    """
    actual_positives = tp + fn
    if actual_positives == 0:
        return 0.0  # Avoid division by zero if there were no actual positives
    return tp / actual_positives


def calculate_f1_score(tp: int, fp: int, tn: int, fn: int) -> float:
    """
    Calculates the F1-score, the harmonic mean of precision and recall.
    2 * (Precision * Recall) / (Precision + Recall)
    Good for balancing precision and recall, especially with imbalanced classes.
    """
    precision = calculate_precision(tp, fp, tn, fn)
    recall = calculate_recall(tp, fp, tn, fn)

    if precision + recall == 0:
        return 0.0  # Avoid division by zero if both precision and recall are 0
    return 2 * (precision * recall) / (precision + recall)


def get_ground_truth_data(ds: Dataset, video_id: int) -> Dict[str, Any]:
    """
    Retrieves ground truth data for a specific video from the dataset.

    Args:
        ds (Dataset): The dataset containing ground truth information.
        video_id (int): The ID of the video to retrieve data for.

    Returns:
        Dict[str,Any]: The ground truth data item corresponding to the specified video ID.
        If no matching item is found, returns the last item in the dataset.
    """
    gt_data_item = None
    for data_idx in range(len(ds)):
        gt_data_item = ds[data_idx]
        if gt_data_item["video_id"] == video_id:
            break
    return gt_data_item


def segment_pred_feedbacks(
    pred_texts: List[str], pred_text_timestamps: List[float]
) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Segments prediction texts and their timestamps based on instructions.

    This function divides a sequence of prediction texts and timestamps into segments,
    where each segment starts with an instruction (text containing "Instruct").

    Args:
        pred_texts (List[str]): List of predicted text feedbacks.
        pred_text_timestamps (List[float]): List of timestamps corresponding
                                            to each text feedback.

    Returns:
        Tuple[List[List[str]], List[List[float]]]: A tuple containing:
            - List of segmented prediction texts,
              where each inner list represents one instruction segment
            - List of segmented prediction timestamps corresponding to the texts
    """
    pred_texts_segmented, pred_text_timestamps_segmented, last_instruction = [], [], 0
    for pred_idx, pred_text in enumerate(pred_texts[1:]):
        if "Instruct" in pred_text:
            pred_texts_segmented.append(
                pred_texts[last_instruction : pred_idx + 1],
            )
            pred_text_timestamps_segmented.append(
                pred_text_timestamps[last_instruction : pred_idx + 1]
            )
            last_instruction = pred_idx + 1

    if last_instruction + 1 < len(pred_texts):  # add any remaining predictions into new segment
        pred_texts_segmented.append(pred_texts[last_instruction:])
        pred_text_timestamps_segmented.append(pred_text_timestamps[last_instruction:])

    return pred_texts_segmented, pred_text_timestamps_segmented


def find_closest_temporal_feedback(
    _pred_timestamp: float,
    gt_texts: List[str],
    gt_text_timestamps: List[float],
    gt_text_types: List[str],
) -> Tuple[int, str, float]:
    """
    Finds the ground truth feedback that is temporally closest to the predicted feedback timestamp.

    Args:
        _pred_timestamp (float): The timestamp of the predicted feedback.
        gt_texts (List[str]): List of ground truth text feedbacks.
        gt_text_timestamps (List[float]): List of timestamps for the ground truth feedbacks.
        gt_text_types (List[str]): List of types for the ground truth feedbacks.

    Returns:
        Tuple[int, str, float]: A tuple containing:
            - The index of the closest feedback in the ground truth lists
            - The text of the closest feedback
            - The timestamp of the closest feedback

    Raises:
        Exception: If no mistakes are found in the ground truth data.
    """
    closest_feedback_idx, closest_feedback_distance = -1, float("inf")
    # Loop over all gt feedbacks to find match
    for feedback_idx in range(len(gt_text_timestamps)):
        feedback_type = gt_text_types[feedback_idx]
        if "mistake" in feedback_type:
            feedback_distance = abs(gt_text_timestamps[feedback_idx] - _pred_timestamp)
            if feedback_distance < closest_feedback_distance and gt_texts[feedback_idx] is not None:
                closest_feedback_distance = feedback_distance
                closest_feedback_idx = feedback_idx

    if closest_feedback_idx >= 0:
        # This is the temporally closest mistake
        gt_closest_text_idx = closest_feedback_idx
        gt_closest_text = gt_texts[closest_feedback_idx]
        gt_closest_text_timestamp = gt_text_timestamps[closest_feedback_idx]
        return gt_closest_text_idx, gt_closest_text, gt_closest_text_timestamp
    else:
        raise Exception("No mistakes found in ground truth data!")


def check_if_correct_instruction_given(
    gt_instruction: str, pred_instruction: str, match_threshold: float = 0.8
) -> bool:
    """
    Determines if a predicted instruction matches the ground truth instruction.

    Args:
        gt_instruction (str): The ground truth instruction text.
        pred_instruction (str): The predicted instruction text to compare.
        match_threshold (float, optional): The minimum ROUGE score required to consider
            the instructions as matching. Defaults to 0.8.

    Returns:
        bool: True if the ROUGE score between the instructions meets or exceeds the
            threshold, False otherwise.
    """
    instruction_rouge_score = get_rouge_score(pred_instruction, gt_instruction)
    return instruction_rouge_score >= match_threshold


def load_json_file_to_dict(file_name: str) -> Dict[Any, Any]:
    """
    Load a JSON file and return its contents as a Python dictionary.

    Args:
        file_name (str): Path to the JSON file to be loaded.

    Returns:
        Dict[Any,Any]: Dictionary containing the data from the JSON file.
    """
    with open(file_name, "r") as f:
        data = json.load(f)
    return data


def run(dataset: Dataset, predictions: Dict[str, Any]) -> None:
    """
    Evaluates model predictions against ground truth data.

    This function processes predictions and compares them with ground truth data to calculate:
    1. Instruction completion accuracy (IC-Acc) - how well the model detects successful instruction completions
    2. Mistake detection metrics (accuracy, precision, recall, F1) - how well the model identifies mistakes
    3. Fluency of mistake feedback using BERT and ROUGE-L scores

    Args:
        dataset: Dataset containing ground truth data with instructions, timestamps, and mistake annotations
        predictions: Dictionary containing model predictions with texts and timestamps

    Returns:
        None: Results are printed to standard output
    """
    num_detected_instructions, total_instructions_without_mistakes = 0, 0
    mistake_tp, mistake_fp, mistake_tn, mistake_fn = 0, 0, 0, 0
    bert_scores, rouge_scores = [], []

    # Eval start
    for eval_idx in tqdm(range(len(predictions))):
        print("=" * 80)
        predictions_item = predictions[eval_idx]

        video_id = predictions_item["video_id"]
        pred_texts = predictions_item["pred_texts"]
        pred_text_timestamps = predictions_item["pred_timestamps"]

        gt_data_item = get_ground_truth_data(dataset, video_id)

        total_instructions_without_mistakes += sum([(not x) for x in gt_data_item["has_mistake"]])

        pred_texts_segmented, pred_text_timestamps_segmented = segment_pred_feedbacks(
            pred_texts, pred_text_timestamps
        )

        #
        for segment_idx in range(min(len(pred_texts_segmented), len(gt_data_item["texts"]))):
            gt_texts = gt_data_item["texts"][segment_idx]
            gt_text_timestamps = gt_data_item["texts_timestamps"][segment_idx]
            gt_text_types = gt_data_item["texts_types"][segment_idx]
            gt_contains_mistake = gt_data_item["has_mistake"][
                segment_idx
            ]  # Does the person make a mistake?

            gt_instruction = gt_data_item["texts"][segment_idx][0]
            gt_instruction_end_time = max(
                gt_data_item["texts_timestamps"][segment_idx]
            )  # When the person completes the instruction w or w/o mistakes

            _segment_pred_texts = pred_texts_segmented[segment_idx]
            _segment_pred_text_timestamps = pred_text_timestamps_segmented[segment_idx]

            _mistake_register = [False] * (len(gt_texts) - 1) if gt_contains_mistake else []

            # If no feedbacks were produced
            if len(_segment_pred_texts[1:]) == 0:
                if gt_contains_mistake:
                    mistake_fn += sum(["mistake" in _type for _type in gt_text_types])
                continue

            # Check if correct instruction provided
            pred_instruction = _segment_pred_texts[0].replace("Instruct:", "").strip()
            is_correct_instruction = check_if_correct_instruction_given(
                gt_instruction, pred_instruction
            )

            if not is_correct_instruction:
                # Incorrect instruction provided
                if gt_contains_mistake:
                    mistake_fn += sum(["mistake" in _type for _type in gt_text_types])
                continue

            # COMPUTE IC-ACC
            if not gt_contains_mistake:
                if (
                    "Success:" in _segment_pred_texts[-1]
                    or "all the steps" in _segment_pred_texts[-1]
                ) and len(_segment_pred_texts) == 2:
                    _pred_instruction_completion_time = _segment_pred_text_timestamps[-1]
                    if (
                        abs(_pred_instruction_completion_time - gt_instruction_end_time)
                        < DETECTION_WINDOW_LENGTH
                    ):
                        num_detected_instructions += 1

            # GET MISTAKE METRICS
            pred_contains_mistake = any(["Feedback" in _text for _text in _segment_pred_texts[1:]])
            for _pred_text, _pred_timestamp in zip(
                _segment_pred_texts[1:], _segment_pred_text_timestamps[1:]
            ):
                if ("Feedback" in _pred_text) and ("all the steps" not in _pred_text):
                    if not gt_contains_mistake:
                        # False positive as no GT mistake present
                        mistake_fp += 1
                    else:
                        # Find the temporally closest mistake feedback
                        gt_closest_text_idx, gt_closest_text, gt_closest_text_timestamp = (
                            find_closest_temporal_feedback(
                                _pred_timestamp, gt_texts, gt_text_timestamps, gt_text_types
                            )
                        )

                        if (
                            abs(_pred_timestamp - gt_closest_text_timestamp)
                            < DETECTION_WINDOW_LENGTH
                        ):
                            _pred_text = _pred_text.replace("Feedback:", "").strip()
                            for _ig_str in IGNORE_STRINGS:
                                _pred_text = _pred_text.replace(_ig_str, "")
                            _pred_text = _pred_text.strip()

                            for _ig_str in IGNORE_STRINGS:
                                gt_closest_text = gt_closest_text.replace(_ig_str, "")
                            gt_closest_text = gt_closest_text.strip()

                            _bert_score = get_bert_score(_pred_text, gt_closest_text)
                            bert_scores.append(_bert_score)

                            _rouge_score = get_rouge_score(_pred_text, gt_closest_text)
                            rouge_scores.append(_rouge_score)

                            # False true positive as a GT mistake present within the DETECTION_WINDOW_LENGTH
                            # Use of _mistake_register to count TP only once
                            _mistake_register[gt_closest_text_idx - 1] = True

                        else:
                            # False positive as no GT mistake present within the DETECTION_WINDOW_LENGTH
                            mistake_fp += 1

            mistake_tn += not gt_contains_mistake and not pred_contains_mistake
            mistake_tp += sum(_mistake_register)
            mistake_fn += sum([not _register for _register in _mistake_register])

        if len(pred_texts_segmented) < len(gt_data_item["texts"]):
            for segment_idx in range(len(pred_texts_segmented), len(gt_data_item["texts"])):
                gt_contains_mistake = gt_data_item["has_mistake"][segment_idx]
                if gt_contains_mistake:
                    # False negative as no GT mistake not detected
                    mistake_fn += sum(
                        ["mistake" in _type for _type in gt_data_item["texts_types"][segment_idx]]
                    )

        if len(gt_data_item["texts"]) < len(pred_texts_segmented):
            for segment_idx in range(len(gt_data_item["texts"]), len(pred_texts_segmented)):
                _segment_pred_texts = pred_texts_segmented[segment_idx]
                for _text in _segment_pred_texts:
                    if "Feedback" in _text:
                        # False positive as no GT mistake present
                        mistake_fp += 1

        print("\n<<<")
        print(f"DETECTION_WINDOW_LENGTH: {DETECTION_WINDOW_LENGTH}")
        print(
            f"IC-Acc: {((num_detected_instructions/total_instructions_without_mistakes))*100:.1f}"
        )
        print(
            f"mistake_tp:{mistake_tp}, mistake_fp:{mistake_fp}, mistake_tn:{mistake_tn}, mistake_fn:{mistake_fn}"
        )
        print(
            f"Mistake Accuracy: {calculate_accuracy(mistake_tp, mistake_fp, mistake_tn, mistake_fn):.2f} -- "
            f"Mistake Precision: {calculate_precision(mistake_tp, mistake_fp, mistake_tn, mistake_fn):.2f} -- "
            f"Mistake Recall: {calculate_recall(mistake_tp, mistake_fp, mistake_tn, mistake_fn):.2f} -- "
            f"Mistake F1_Score: {calculate_f1_score(mistake_tp, mistake_fp, mistake_tn, mistake_fn):.2f} -- "
        )
        print(
            f"Mistake Mean BERT Score: {mean(bert_scores):.3f}, Mistake Mean Rouge-L Score: {mean(rouge_scores):.3f}"
        )
        print(">>>\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    dataset = load_dataset(args.plan_set, args.split)
    predictions = load_json_file_to_dict(args.predictions_file_path)
    run(dataset, predictions)
