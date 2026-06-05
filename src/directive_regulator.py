#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   directive_regulator.py                               :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: kmalfois <kmalfois@student.42.fr>            +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/05/21 14:48:39 by kmalfois            #+#    #+#            #
#   Updated: 2026/06/05 14:49:24 by kmalfois           ###   ########.fr      #
#                                                                             #
# ########################################################################### #


import os
# import sys
import json
from typing import Any
from llm_sdk import Small_LLM_Model as llm_model


class DirectiveRegulator():
    def __init__(
            self,
            func_definitions: list[dict[str, Any]],
            prompts: list[dict[str, str]],
            model: llm_model
    ) -> None:
        self.user_prompts = prompts
        self.func_definitions: str = json.dumps(func_definitions, indent=2)
        self.func_data: list = [
            {"name": f['name'], "parameters": f['parameters']} for f in func_definitions
        ]
        self.model: llm_model = model
        self.prompt_nbr = 0

    def __str__(self):
        return self.__directive

    def load_directive(self, prompt_nbr: int) -> None:
        self.prompt_nbr = prompt_nbr
        self.directive = (
            "You are a precise function selection assistant.\n"
            "Choose exactly one tool from the list below to fulfill"
            "the user request.\n"
            "Respond or complete ONLY with a VALID raw JSON string "
            "without any identations.\n\n"
            f"Available functions: {self.func_definitions}"
            f"User Prompt: {self.user_prompts[prompt_nbr]}"
            "Response:"
        )
        # print(f"\033[31m{self.user_prompts[prompt_nbr]}\033[0m")

    @property
    def directive(self) -> str:
        return self.__directive

    @directive.setter
    def directive(self, dir: str) -> None:
        self.__directive: str = dir

    @property
    def user_prompts(self) -> list[str]:
        return self.__user_prompts

    @user_prompts.setter
    def user_prompts(
            self,
            prompts: list[dict[str, str]]
    ) -> None:
        usr_prmpt = []
        for prompt in prompts:
            key = next(iter(prompt))
            usr_prmpt.append(prompt[key])
        self.__user_prompts: list[str] = usr_prmpt

    def encoding(self) -> list[int]:
        return self.model.encode(self.directive).tolist()[0]

    def prompts_count(self) -> int:
        return len(self.user_prompts)

    def current_usr_prompt(self) -> str:
        prompt = self.user_prompts[self.prompt_nbr]
        converted_prompt = json.dumps(prompt)[1:-1]
        return converted_prompt
