#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   llm_cache.py                                         :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: kmalfois <kmalfois@student.42.fr>            +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/10 09:24:04 by kmalfois            #+#    #+#            #
#   Updated: 2026/06/12 17:08:21 by kmalfois           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import torch
from llm_sdk import Small_LLM_Model as llm_model


class CachedLLM:
    def __init__(self, llm: llm_model) -> None:
        self.llm = llm
        self.model = llm._model
        self.device = llm._device
        self.past_key_values = None
        self.cached_input_length = 0

    def reset_cache(self):
        self.past_key_values = None
        self.cached_input_length = 0

    def get_logits_cached(self, response_ids: list[int]) -> list[float]:
        if len(response_ids) <= self.cached_input_length:
            self.reset_cache()
        if self.past_key_values is None:
            input_tensor = torch.tensor([response_ids], device=self.device, dtype=torch.long)
            with torch.no_grad():
                out = self.model(
                    input_ids=input_tensor,
                    past_key_values=None,
                    use_cache=True
                )
            self.past_key_values = out.past_key_values
            self.cached_input_length = len(response_ids)
            logits = out.logits[0, -1].tolist()
            return [float(x) for x in logits]
        else:
            new_tokens = response_ids[self.cached_input_length:]
            if not new_tokens:
                return self.llm.get_logits_from_input_ids(response_ids)
            input_tensor = torch.tensor([new_tokens], device=self.device, dtype=torch.long)
            with torch.no_grad():
                out = self.model(
                    input_ids=input_tensor,
                    past_key_values=self.past_key_values,
                    use_cache=True
                )
            self.past_key_values = out.past_key_values
            self.cached_input_length += len(new_tokens)
            logits = out.logits[0, -1].tolist()
            return [float(x) for x in logits]

    def encode(self, text: str) -> list[int]:
        return self.llm.encode(text)

    def decode(self, list_ids: list[int]) -> str:
        return self.llm.decode(list_ids)
