#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   crucible.py                                          :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: kmalfois <kmalfois@student.42.fr>            +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/05/22 12:37:07 by kmalfois            #+#    #+#            #
#   Updated: 2026/06/12 17:08:19 by kmalfois           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import re
import json
import time
from typing import Any
from llm_sdk import Small_LLM_Model as llm_model
from src.llm_cache import CachedLLM as cachedllm
from src.json_validator import OutputValidator as ov
from src.lexicon import Lexicon as lex
from src.directive_regulator import DirectiveRegulator as dr
from src.guardrail import GuardRail as gr
from src.guardrail import State


class Crucible():
    def __init__(
            self,
            user_prompts: list[dict[str, Any]],
            func_definitions: list[dict[str, Any]],
    ) -> None:
        raw_model = llm_model()
        self.model = cachedllm(raw_model)
        self.lexicon = lex(raw_model)
        self.directive: dr = dr(
            func_definitions,
            user_prompts,
            self.lexicon
        )
        self.responses: list[str] = []
        self.response_ids: list[int] = []
        self.guardrail = gr(
            self.model,
            self.directive,
            self.response_ids,
            self.lexicon
        )

    def __str__(self) -> str:
        return (
            "Current response string:\n"
            f"\033[35m{self.responses}\033[0m"
        )

    def response_gen(self) -> None:
        for prompt_nbr in range(self.directive.prompts_count()):
            self.directive.load_directive(prompt_nbr)
            response = self.llm_processing()
            self.responses.append(response)
        print(self.responses)
        try:
            validated_json = ov.output_validator(self.responses)
        except Exception:
            print("/!\\ Output Validation went wrong")
        json_to_write = json.dumps([record.model_dump() for record in validated_json], indent=4)
        print(f"\033[35m{json_to_write}\033[0m")

    def llm_processing(self) -> str:
        # print(f"\n--- LLM Processing {len(self.response_ids)} Check ---")
        # t = time.perf_counter()
        context_ids: list[int] = self.directive.encoding()
        # t = self.profile_section("LLM Processing ids encoding", t)
        self.response_ids.clear()
        self.model.reset_cache()  # llm_cache related
        # t = self.profile_section("LLM Processing response_ids clear", t)
        self.guardrail.state = State.OPEN_JSON
        while self.finished_check(self.response_ids) is False:
            # t = self.profile_section("LLM Processing loop start", t)
            # logits = self.model.get_logits_from_input_ids(
            #     context_ids + self.response_ids)
            logits = self.model.get_logits_cached(
                context_ids + self.response_ids)
            # t = self.profile_section("LLM Processing logit recovery", t)
            token_id = self.guardrail.logit_sorter(logits)
            # t = self.profile_section("LLM Processing logit sorting", t)
            self.response_ids.append(token_id)
            self.guardrail.update_state()
            # t = self.profile_section("LLM Processing GR state update", t)
            print(f"{self.lexicon.lex_decode(self.response_ids)}")
            print(f"\033[36m{self.guardrail.state} | "
                  f"'{self.guardrail.function}' | "
                  f"{self.guardrail.schema_stack}\033[0m\n\n")
        self.guardrail.guardrail_clear()
        if not self.response_ids:
            return "{}"
        raw_string = self.lexicon.lex_decode(self.response_ids)
        clean_string = raw_string.replace('\u0120', ' ').replace('\u010a', '\n')
        print(f"\033[32m{clean_string}\033[0m")
        return clean_string

    def finished_check(self, response_ids: list[int]) -> bool:
        if not response_ids:
            return False
        if self.guardrail.state != State.CLOSE_JSON:
            return False
        response_txt: str = self.lexicon.lex_decode(response_ids)
        cleaned_txt = re.sub(r'"(\\.|[^"\\])*"', '', response_txt)
        return cleaned_txt.count("{") == cleaned_txt.count("}")

    def build_dictionary(self) -> None:
        # llm_dict = self.model.get_path_to_vocab_file()
        llm_dict = self.model.llm.get_path_to_vocab_file()
        with open(llm_dict, "r", encoding="utf-8") as vocab_file:
            vocabulary = json.load(vocab_file)
        self.token_to_id: dict[str, int] = vocabulary
        id_to_token = {int(v): k for k, v in vocabulary.items()}
        self.id_to_token: dict[int, str] = id_to_token

    # CHRONO WRAPPER
    def profile_section(self, name: str, start_time: float) -> float:
        """Prints the elapsed time for a section and returns the new baseline time."""
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        print(f"   [TIMER] {name}: {duration_ms:.4f} ms")
        return end_time
