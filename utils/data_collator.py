from dataclasses import dataclass
from typing import List, Union, Any, Dict, Optional
from collections.abc import Mapping

from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.data.data_collator import _torch_collate_batch


@dataclass
class CustomDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator that:
    - Dynamically pads inputs to the max length in the batch.
    - Uses existing 'labels' key in the batch if present.
    - Otherwise, for causal LM (mlm=False), clones input_ids to labels,
      replacing pad tokens with -100.
    """
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = False
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(
        self,
        examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        # 1) Pad inputs and convert to torch tensors
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                examples,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(
                    examples,
                    self.tokenizer,
                    pad_to_multiple_of=self.pad_to_multiple_of
                )
            }

        # 2) Remove special_tokens_mask if present
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        if self.mlm:
            # 3a) For masked LM, apply masking and generate labels
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"],
                special_tokens_mask=special_tokens_mask
            )
        else:
            # 3b) For causal LM, preserve existing labels or clone input_ids
            if "labels" not in batch:
                labels = batch["input_ids"].clone()
                # Mark pad tokens as -100 so they are ignored by the loss
                if self.tokenizer.pad_token_id is not None:
                    labels[labels == self.tokenizer.pad_token_id] = -100
                batch["labels"] = labels

        return batch
