#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   crucible.py                                          :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: kmalfois <kmalfois@student.42.fr>            +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/05/22 12:37:07 by kmalfois            #+#    #+#            #
#   Updated: 2026/06/05 14:30:09 by kmalfois           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
import os
from typing import Any
from llm_sdk import Small_LLM_Model as llm_model
from src.directive_regulator import DirectiveRegulator as dr
from src.guardrail import GuardRail as gr
from src.guardrail import State


class Crucible():
    def __init__(
            self,
            user_prompts: list[dict[str, Any]],
            func_definitions: list[dict[str, Any]],
    ) -> None:
        self.model = llm_model()
        self.build_dictionary()
        self.directive: dr = dr(
            func_definitions,
            user_prompts,
            self.model
        )
        self.responses: list[str] = []
        self.response_ids: list[int] = []
        self.guardrail = gr(self.model, self.directive, self.response_ids, self.dictionary)

    def __str__(self) -> str:
        return (
            "Current response string:\n"
            f"\033[35m{json.dumps(self.responses, indent=4)}\033[0m"
        )

    def response_gen(self) -> None:
        for prompt_nbr in range(self.directive.prompts_count()):
            self.directive.load_directive(prompt_nbr)
            response = self.llm_processing()
            self.responses.append(response)

    def llm_processing(self) -> str:
        context_ids: list[int] = self.directive.encoding()
        self.response_ids.clear()
        self.guardrail.state = State.OPEN_JSON
        while self.finished_check(self.response_ids) is False:
            logits = self.model.get_logits_from_input_ids(
                context_ids + self.response_ids)
            token_id = self.guardrail.logit_sorter(logits)
            self.response_ids.append(token_id)
            self.guardrail.update_state()
            if self.guardrail.function == "":
                self.guardrail.recover_func_name()
            print(f"{self.model.decode(self.response_ids)}")
            print(f"\033[36m{self.guardrail.state} | "
                  f"'{self.guardrail.function}' | "
                  f"{self.guardrail.schema_stack}\033[0m\n\n")
        self.guardrail.clear_guardrail()
        if not self.response_ids:
            return "{}"
        raw_string = self.model.decode(self.response_ids)
        clean_string = raw_string.replace('\u0120', ' ').replace('\u010a', '\n')
        parsed_json = json.loads(clean_string)
        print(f"\033[32m{json.dumps(parsed_json, indent=4)}")
        return parsed_json

    def finished_check(self, response_ids: list[int]) -> bool:
        if not response_ids:
            return False
        response_txt: str = self.model.decode(response_ids)
        return response_txt.count("{") == response_txt.count("}")

    def build_dictionary(self) -> None:
        llm_dict = self.model.get_path_to_vocab_file()
        with open(llm_dict, "r", encoding="utf-8") as vocab_file:
            vocabulary = json.load(vocab_file)
        vocab_map = {int(v): k for k, v in vocabulary.items()}
        self.dictionary: dict[int, str] = vocab_map
