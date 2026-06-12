#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   json_validator.py                                    :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: kmalfois <kmalfois@student.42.fr>            +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/05/20 11:31:47 by kmalfois            #+#    #+#            #
#   Updated: 2026/06/12 17:08:20 by kmalfois           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
import re
from pydantic import BaseModel, Field, ValidationError
from typing import Any, Optional



class UserPromptValidator(BaseModel):
    prompt: str = Field(..., alias="prompt")

    @classmethod
    def prompt_validator(cls, list_prompts: list[dict[str, str]]) -> list["UserPromptValidator"]:
        validated_prompts: list[UserPromptValidator] = []

        for index, prompt in enumerate(list_prompts):
            try:
                valid_record = cls.model_validate(prompt)
                validated_prompts.append(valid_record)
            except ValidationError as e:
                print(f"\033[31m Error in prompt {index} format")
                print(f"{e}\033[0m")
        print("\033[32mPrompts format validated\033[0m")
        return validated_prompts


class FunctionValidator(BaseModel):
    name: str = Field(..., alias="name")
    description: str = Field(..., alias="description")
    parameters: dict[str, "FunctionType"] = Field(..., alias="parameters")
    returns: "FunctionType" = Field(..., alias="returns")

    @classmethod
    def function_validator(cls, list_functions: list[dict]) -> list["FunctionValidator"]:
        validated_functions: list[FunctionValidator] = []
        is_valid = True
        for index, function in enumerate(list_functions):
            try:
                valid_record = cls.model_validate(function)
                validated_functions.append(valid_record)
            except ValidationError as e:
                print(f"\033[31m Error in function {index} format")
                print(f"{e}\033[0m")
        if is_valid:
            print("\033[32mFunctions format validated\033[0m")
        else:
            raise ValidationError
        return validated_functions


class FunctionType(BaseModel):
    type: str = Field(..., alias="type", description="Parameters & Retruns type")
    parameters: Optional[dict[str, Any]] = Field(None, alias="parameters")
    model_config = {"extra": "forbid"}


class OutputValidator(BaseModel):
    prompt: str = Field(..., alias="prompt")
    name: str = Field(..., alias="name")
    parameters: dict[str, Any] = Field(..., alias="parameters")

    @classmethod
    def output_validator(cls, list_responses: list[str]) -> list["OutputValidator"]:
        validated_responses: list[OutputValidator] = []

        for index, response in enumerate(list_responses):
            cleared_response = OutputValidator.sanatize_json_escapes(response)
            try:
                valid_record = cls.model_validate_json(cleared_response)
                validated_responses.append(valid_record)
                print(f"\033[32mPrompt {index} resolution validated\033[0m")
            except ValidationError as e:
                print(f"\033[31m Error in prompt {index} resolution")
                print(f"{e}\033[0m")

        return validated_responses

    @staticmethod
    def sanatize_json_escapes(raw_string: str) -> str:
        return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw_string)
