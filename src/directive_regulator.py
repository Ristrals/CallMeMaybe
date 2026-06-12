#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   directive_regulator.py                               :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: kmalfois <kmalfois@student.42.fr>            +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/05/21 14:48:39 by kmalfois            #+#    #+#            #
#   Updated: 2026/06/12 17:08:18 by kmalfois           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
from typing import Any
from src.lexicon import Lexicon as lex
from src.json_validator import UserPromptValidator as upv
from src.json_validator import FunctionValidator as fv


class DirectiveRegulator():
    def __init__(
            self,
            func_definitions: list[dict[str, Any]],
            prompts: list[dict[str, str]],
            lexicon: lex
    ) -> None:
        self.func_definitions: list[fv] = fv.function_validator(func_definitions)
        self.user_prompts: list[upv] = upv.prompt_validator(prompts)
        self.func_data: list = [
            {"name": f['name'], "parameters": f['parameters']} for f in func_definitions
        ]
        self.lexicon = lexicon
        self.prompt_nbr = 0

    def __str__(self):
        return self.__directive

    def load_directive(self, prompt_nbr: int) -> None:
        self.prompt_nbr = prompt_nbr
        self.directive = (
            "/no_think\n"
            "You are a precise function selection assistant.\n"
            "Choose exactly one tool from the list below to fulfill"
            "the user request.\n"
            "Respond or complete ONLY with a VALID raw JSON string.\n"
            "CRITICAL: All nested parameters containing regular expression symbols must have properly escaped strings. Do not use naked $ macros.\n"
            "NO identations of any type.\n\n"
            f"Available functions: {self.func_definitions}"
            f"User Prompt: {self.user_prompts[prompt_nbr].prompt}"
            "Response:"
        )
        # print(f"\033[31m{self.user_prompts[prompt_nbr].prompt}\033[0m")

    @property
    def directive(self) -> str:
        return self.__directive

    @directive.setter
    def directive(self, directive: str) -> None:
        self.__directive: str = directive

    @property
    def user_prompts(self) -> list[upv]:
        return self.__user_prompts

    @user_prompts.setter
    def user_prompts(
            self,
            prompts: list[upv]
    ) -> None:
        self.__user_prompts: list[upv] = prompts

    def encoding(self) -> list[int]:
        return self.lexicon.lex_encode(self.directive)

    def prompts_count(self) -> int:
        return len(self.user_prompts)

    def current_usr_prompt(self) -> str:
        prompt = self.user_prompts[self.prompt_nbr].prompt
        converted_prompt = json.dumps(prompt)[1:-1]
        return converted_prompt
