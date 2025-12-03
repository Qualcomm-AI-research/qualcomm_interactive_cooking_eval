# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from typing import Dict, List, Union

from datasets import load_dataset
from torch.utils.data import Dataset


class QualcommInteractiveCookingDataset(Dataset):
    """
    A dataset class for the Qualcomm Interactive Cooking dataset.

    Args:
        plan_set (str): Type of planning set, either "main" or "advanced"
        split (str): Dataset split, one of "train", "validation", or "test"
    """

    hf_dataset_name = "qualcomm/qualcomm-interactive-cooking-dataset"

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
