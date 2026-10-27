# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from typing import Dict, List, Optional, Union
import os
import json

from datasets import load_dataset
from torch.utils.data import Dataset


class QualcommInteractiveCookingDatasetNeurIPS2025(Dataset):
    """
    A dataset class for the Qualcomm Interactive Cooking dataset.

    Args:
        plan_set (str): Type of planning set, either "main" or "advanced"
        split (str): Dataset split, one of "train", "validation", or "test"
    """
    hf_dataset_name = "qualcomm/qualcomm-interactive-cooking-dataset"
    __version__ = "1.0"

    def __init__(self, plan_set: str, split: str) -> None:
        assert plan_set in ["main", "advanced_planning"]
        assert split in ["train", "validation", "test"]

        self.plan_set = plan_set
        self.split = split

        self.load_data()

    def load_data(self) -> None:
        """
        Load data from huggingface based on plan_set and split.
        """
        if self.plan_set == "main":
            if self.split == "train":
                self.annotations = load_dataset(
                    self.hf_dataset_name, "main", split="train"
                ).to_list()
                self.annotations += load_dataset(
                    self.hf_dataset_name, "main", split="validation"
                ).to_list()
            elif self.split == "validation":
                self.annotations = load_dataset(
                    self.hf_dataset_name, "main", split="validation"
                ).to_list()
            elif self.split == "test":
                self.annotations = load_dataset(
                    self.hf_dataset_name, "main", split="test"
                ).to_list()
            else:
                raise ValueError(f"Invalid Split: {self.split}")

        elif self.plan_set == "advanced_planning":
            if self.split == "train":
                self.annotations = load_dataset(
                    self.hf_dataset_name, "advanced_planning", split="train"
                ).to_list()
                self.annotations += load_dataset(
                    self.hf_dataset_name, "advanced_planning", split="validation"
                ).to_list()
            elif self.split == "validation":
                self.annotations = load_dataset(
                    self.hf_dataset_name, "advanced_planning", split="validation"
                ).to_list()
            elif self.split == "test":
                self.annotations = load_dataset(
                    self.hf_dataset_name, "advanced_planning", split="test"
                ).to_list()
            else:
                raise ValueError(f"Invalid Split: {self.split}")
        else:
            raise ValueError(f"Invalid Plan Set: {self.plan_set}")

        self.preprocess_data()

    def preprocess_data(self) -> None:
        """
        Preprocess the loaded annotations.
        """
        for annotation in self.annotations:
            output_texts = annotation["output_texts"]
            output_timestamps = annotation["output_timestamps"]
            output_types = annotation["output_types"]
            # Ignore any dangling instructions
            if not ("finish_all" in output_types[-1]):
                if output_types[-1] == "instruction":
                    for k in [
                        "remaining_plan",
                        "output_timestamps",
                        "output_texts",
                        "output_types",
                        "output_actions",
                    ]:
                        annotation[k] = annotation[k][:-1]

            (
                instruction_segment_texts,
                instruction_segment_texts_timestamps,
                instruction_segment_texts_types,
            ) = ([], [], [])
            instrcution_segment_has_mistake = []

            _segment_texts, _segment_texts_timestamps, _segment_texts_types = [], [], []
            has_mistake = False
            for _text, _timestamp, _type in zip(output_texts, output_timestamps, output_types):
                if _type == "instruction" or "finish_all" in _type:
                    if len(_segment_texts) > 0:
                        instruction_segment_texts.append(_segment_texts)
                        instruction_segment_texts_timestamps.append(_segment_texts_timestamps)
                        instruction_segment_texts_types.append(_segment_texts_types)
                        instrcution_segment_has_mistake.append(has_mistake)

                    _segment_texts = [_text]
                    _segment_texts_timestamps = [_timestamp]
                    _segment_texts_types = [_type]
                    has_mistake = False
                else:
                    _segment_texts.append(_text)
                    _segment_texts_timestamps.append(_timestamp)
                    _segment_texts_types.append(_type)
                    if "mistake" in _type:
                        has_mistake = True

            annotation.update(
                {
                    "instruction_segment_texts": instruction_segment_texts,
                    "instruction_segment_texts_timestamps": instruction_segment_texts_timestamps,
                    "instruction_segment_texts_types": instruction_segment_texts_types,
                    "instrcution_segment_has_mistake": instrcution_segment_has_mistake,
                }
            )

    def __len__(self) -> int:
        """
        Return the number of annotations in the dataset.

        Returns:
            int: Number of annotations
        """
        return len(self.annotations)

    def __getitem__(self, video_idx: int) -> Dict[str, Union[str, List[str]]]:
        """
        Get a specific item from the dataset.

        Args:
            video_idx (int): Index of the video to retrieve

        Returns:
            dict: Dictionary containing video ID, texts, timestamps, types, and mistake indicators
        """
        video_idx = video_idx % len(self.annotations)
        return_dict = {
            "video_id": self.annotations[video_idx]["video_id"],
            "texts": self.annotations[video_idx]["instruction_segment_texts"],
            "texts_timestamps": self.annotations[video_idx]["instruction_segment_texts_timestamps"],
            "texts_types": self.annotations[video_idx]["instruction_segment_texts_types"],
            "has_mistake": self.annotations[video_idx]["instrcution_segment_has_mistake"],
        }

        return return_dict


class EgoMCBench(Dataset):
    """
    A dataset class for the HF-hosted Qualcomm Interactive Cooking dataset.

    Args:
        split (str): Dataset split to load. Defaults to "test".
    """
    hf_dataset_name = "qualcomm/qualcomm-interactive-cooking-dataset-ego-mistake-corrections"
    hf_cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "datasets_qicd_v3"
    )
    __version__ = "3.0"

    def __init__(self, split: str = "test", cache_dir: Optional[str] = None) -> None:
        self.split = split
        self.cache_dir = cache_dir or self.hf_cache_dir
        self.load_data()
        self.preprocess_data()

    def preprocess_data(self) -> None:
        """
        Preprocess the loaded annotations.
        """
        for annotation in self.annotations:
            output_texts = annotation["output_texts"]
            output_timestamps = annotation["output_timestamps"]
            output_types = annotation["output_types"]
            # Ignore any dangling instructions
            if not ("finish_all" in output_types[-1]):
                if output_types[-1] == "instruction":
                    for k in [
                        "remaining_plan",
                        "output_timestamps",
                        "output_texts",
                        "output_types",
                        "output_actions",
                    ]:
                        annotation[k] = annotation[k][:-1]

            (
                instruction_segment_texts,
                instruction_segment_texts_timestamps,
                instruction_segment_texts_types,
            ) = ([], [], [])
            instrcution_segment_has_mistake = []

            _segment_texts, _segment_texts_timestamps, _segment_texts_types = [], [], []
            has_mistake = False
            for _text, _timestamp, _type in zip(output_texts, output_timestamps, output_types):
                if (_type == "instruction") or ("finish_all" in _type):
                    if len(_segment_texts) > 0:
                        instruction_segment_texts.append(_segment_texts)
                        instruction_segment_texts_timestamps.append(_segment_texts_timestamps)
                        instruction_segment_texts_types.append(_segment_texts_types)
                        instrcution_segment_has_mistake.append(has_mistake)

                    _segment_texts = [_text]
                    _segment_texts_timestamps = [_timestamp]
                    _segment_texts_types = [_type]
                    has_mistake = False
                else:
                    _segment_texts.append(_text)
                    _segment_texts_timestamps.append(_timestamp)
                    _segment_texts_types.append(_type)
                    if "mistake" in _type:
                        has_mistake = True

            
            annotation.update(
                {
                    "instruction_segment_texts": instruction_segment_texts,
                    "instruction_segment_texts_timestamps": instruction_segment_texts_timestamps,
                    "instruction_segment_texts_types": instruction_segment_texts_types,
                    "instrcution_segment_has_mistake": instrcution_segment_has_mistake,
                }
            )

    def load_data(self) -> None:
        """
        Load data directly from Hugging Face and convert it to the V2 annotation format.
        """
        _annotations = load_dataset(
            self.hf_dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
        ).to_list()

        self.annotations = []
        _num_of_instructions, _num_of_feedbacks = 0, 0
        _num_of_feedback_replies, _num_of_instruction_clarifications = 0, 0
        _num_of_user_responses = 0
        for annotation in _annotations:
            output_texts, output_text_types, output_text_timestamps = [], [], []
            video_id = annotation["video_id"]
            for _text, _text_type, _timestamp in zip(
                annotation["texts"],
                annotation["text_types"],
                annotation["text_timestamps"],
            ):
                _add_flag = True
                if _text_type == "Instruction":
                    _type = "instruction"
                    _num_of_instructions += 1
                elif _text_type == "Feedback":
                    _type = "mistake"
                    _num_of_feedbacks += 1
                elif _text_type == "Feedback-Reply":
                    _type = "feedback_reply"
                    _num_of_feedback_replies += 1
                    _add_flag = False
                elif _text_type == "Instruction-Clarification":
                    _type = "instruction_clarification"
                    _num_of_instruction_clarifications += 1
                    _add_flag = False
                elif _text_type == "Success":
                    _type = "success"
                elif _text_type in ["User-Comment", "User-Question"]:
                    _type = "user_response"
                    _num_of_user_responses += 1
                    _add_flag = False
                else:
                    print(f"_text: {_text}")
                    _type = "user_response"
                    _num_of_user_responses += 1
                    _add_flag = False

                if _add_flag:
                    output_texts.append(_text)
                    output_text_types.append(_type)
                    output_text_timestamps.append(_timestamp)

            if output_text_types:
                output_text_types[-1] = "finish_all"
            self.annotations.append({
                "video_id": video_id,
                "output_texts": output_texts,
                "output_types": output_text_types,
                "output_timestamps": output_text_timestamps,
            })

        print(f"num_of_instructions: {_num_of_instructions}")
        print(f"num_of_feedbacks: {_num_of_feedbacks}")
        print(f"num_of_feedback_replies: {_num_of_feedback_replies}")
        print(f"num_of_instruction_clarifications: {_num_of_instruction_clarifications}")
        print(f"num_of_user_responses: {_num_of_user_responses}")

    def __len__(self) -> int:
        """
        Return the number of annotations in the dataset.

        Returns:
            int: Number of annotations
        """
        return len(self.annotations)

    def __getitem__(self, video_idx: int) -> Dict[str, Union[str, List[str]]]:
        """
        Get a specific item from the dataset.

        Args:
            video_idx (int): Index of the video to retrieve

        Returns:
            dict: Dictionary containing video ID, texts, timestamps, types, and mistake indicators
        """
        video_idx = video_idx % len(self.annotations)
        return_dict = {
            "video_id": self.annotations[video_idx]["video_id"],
            "texts": self.annotations[video_idx]["instruction_segment_texts"],
            "texts_timestamps": self.annotations[video_idx]["instruction_segment_texts_timestamps"],
            "texts_types": self.annotations[video_idx]["instruction_segment_texts_types"],
            "has_mistake": self.annotations[video_idx]["instrcution_segment_has_mistake"],
        }

        return return_dict

