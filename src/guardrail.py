#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   guardrail.py                                         :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: kmalfois <kmalfois@student.42.fr>            +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/05/27 11:49:53 by kmalfois            #+#    #+#            #
#   Updated: 2026/06/05 17:11:27 by kmalfois           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
import os
import numpy as np
import numpy.typing as npt
from types import NoneType
from typing import Any
from enum import Enum
from functools import singledispatchmethod
from src.directive_regulator import DirectiveRegulator as dr
from llm_sdk import Small_LLM_Model as llm_model


class GuardRail():
    def __init__(
            self,
            model: llm_model,
            directive: dr,
            response_ids: list[int],
            dictionary: dict[int, str]
    ) -> None:
        self.model: llm_model = model
        self.directive: dr = directive
        self.response_ids: list[int] = response_ids
        self.response_text: str = ""
        self.dictionary = dictionary
        self.state: Enum = State.OPEN_JSON
        self.logits: npt.ArrayLike = []
        self.mask: npt.ArrayLike = []
        self.function: str = ""
        self.schema_stack: list[list[dict, list, int]] = []
        self.current_key: str = ""
        self.current_type: str = ""

    def logit_sorter(
            self,
            logit_list: list[float]
    ) -> int:
        self.logits = np.array(logit_list)
        self.update_response_text()

        if self.state == State.OPEN_JSON:
            self.mask_blacklist(None)
            self.mask_whitelist(Mask.MASK_START)
        elif self.state == State.EXPECTING_PROMPT:
            self.mask_blacklist(None)
            self.mask_whitelist(Mask.MASK_PROMPT_KEY)
            if self.response_text.endswith('"prompt":'):
                self.mask_whitelist({'\u0120"'})
        elif self.state == State.INSIDE_PROMPT:
            self.mask_blacklist(None)
            self.mask_whitelist(self.directive.current_usr_prompt())
            if self.response_text.endswith(self.directive.current_usr_prompt()):
                self.mask_blacklist(None)
                self.mask_whitelist(Mask.MASK_CLOSEPRMPT)
        elif self.state == State.EXPECTING_NAME:
            self.mask_blacklist(None)
            self.mask_whitelist(Mask.MASK_NAME_KEY)
            if self.response_text.endswith('"name":'):
                self.mask_whitelist({'\u0120"'})
        elif self.state == State.INSIDE_FUNCTION:
            func_list = [f["name"] for f in self.directive.func_data]
            self.mask_blacklist(None)
            self.mask_whitelist(func_list)
            if self.response_text.endswith(tuple(f for f in func_list)):
                self.mask_blacklist(None)
                self.mask_whitelist(Mask.MASK_CLOSEPRMPT)
        elif self.state == State.EXPECTING_PARAM:
            self.mask_blacklist(None)
            self.mask_whitelist(Mask.MASK_PARAM_KEY)
            if self.response_text.endswith('"parameters":'):
                self.mask_whitelist({'\u0120{"'})
        elif self.state == State.EXPECTING_P_KEY:
            self.guardrail_parameter()
        elif self.state == State.INSIDE_PARAM:
            self.guardrail_parameter()
        elif self.state == State.CLOSE_JSON:
            self.mask_blacklist(None)
            self.mask_whitelist(Mask.MASK_END)

        self.logits = self.logits + self.mask
        print(f"\033[33m{int(np.argmax(self.logits))} | "
              f"{self.model.decode([int(np.argmax(self.logits))])} | "
              f"{self.current_key}\033[0m")
        return int(np.argmax(self.logits))

    def update_state(self) -> None:
        self.update_response_text()
        if (self.state == State.OPEN_JSON and self.response_text.endswith('{"')):
            self.state = State.EXPECTING_PROMPT
        elif (self.state == State.EXPECTING_PROMPT and self.response_text.endswith('"prompt":"')):
            self.state = State.INSIDE_PROMPT
        elif (self.state == State.INSIDE_PROMPT and self.response_text.endswith('","')):
            self.state = State.EXPECTING_NAME
        elif (self.state == State.EXPECTING_NAME and self.response_text.endswith('"name":"')):
            self.state = State.INSIDE_FUNCTION
        elif (self.state == State.INSIDE_FUNCTION and self.response_text.endswith('","')):
            self.state = State.EXPECTING_PARAM
        elif (self.state == State.EXPECTING_PARAM and self.response_text.endswith('"parameters":{"')):
            self.state = State.EXPECTING_P_KEY
        elif (self.state == State.EXPECTING_P_KEY and (
            self.response_text.endswith('":"') or
            self.response_text.endswith('":') or
            self.response_text.endswith('":{"')
        )):
            print("\033[31m Scenario 1\033[0m")
            self.state = State.INSIDE_PARAM
        elif (self.state == State.EXPECTING_P_KEY and self.response_text.endswith('"}')):
            print("\033[31m Scenario 2\033[0m")
            if len(self.schema_stack) > 1:
                self.schema_stack.pop()
                self.update_schema()
                schema, keylist, index = self.schema_stack[-1]
                if index < len(keylist):
                    self.state = State.EXPECTING_P_KEY
            else:
                self.state = State.CLOSE_JSON
        elif (self.state == State.INSIDE_PARAM and self.response_text.endswith(',"')):
            print("\033[31m Scenario 3\033[0m")
            self.schema_stack[-1][2] += 1
            schema, keylist, index = self.schema_stack[-1]
            print(f"\033[31m{len(self.schema_stack)} | {index} | {len(keylist)}\033[0m")
            if len(self.schema_stack) > 1 and index >= len(keylist):
                print("\033[31m in pop\033[0m")
                self.schema_stack.pop()
                self.update_schema()
            else:
                print("\033[31m in next\033[0m")
                self.update_schema()
            self.state = State.EXPECTING_P_KEY
        elif (self.state == State.INSIDE_PARAM and (
            self.response_text.endswith('"}') or
            self.response_text.endswith('}')
        )):
            print("\033[31m Scenario 4\033[0m")
            if len(self.schema_stack) > 1:
                self.schema_stack.pop()
                self.update_schema()
                schema, keylist, index = self.schema_stack[-1]
                if index < len(keylist):
                    self.state = State.EXPECTING_P_KEY
                else:
                    self.state = State.CLOSE_JSON
            else:
                self.state = State.CLOSE_JSON

    def recover_func_name(self) -> None:
        self.update_response_text()
        for function in self.directive.func_data:
            if self.response_text.find(function['name']) != -1:
                self.function = function['name']
                param_schema: dict = function.get('parameters', {})
                param_keylist: list = list(param_schema.keys())
                self.schema_stack.append([param_schema, param_keylist, 0])
                self.update_schema()

    def update_schema(self) -> None:
        if self.schema_stack:
            schema, keylist, index = self.schema_stack[-1]
            if 0 <= index < len(keylist):
                self.current_key = keylist[index]
                self.current_type = schema[self.current_key]['type']
            else:
                self.current_key = ""
                self.current_type = ""

    def guardrail_parameter(self) -> None:
        print(self.current_type)
        if self.state == State.EXPECTING_P_KEY:
            self.mask_blacklist(None)
            self.mask_whitelist(self.current_key)
            if self.response_text.endswith(self.current_key):
                if self.current_type == 'number' or self.current_type == 'boolean':
                    self.mask_whitelist(Mask.MASK_CLOSE_PARAM_KEY_NUM)
                elif self.current_type == 'string':
                    self.mask_whitelist(Mask.MASK_CLOSE_PARAM_KEY_STR)
                elif self.current_type == 'function':
                    self.mask_whitelist({'":{"'})
        elif self.state == State.INSIDE_PARAM:
            self.update_schema()
            if self.current_type == "string":
                print("str")
                self.guardrail_string()
            elif self.current_type == "number":
                print("nbr")
                self.guardrail_number()
            elif self.current_type == "boolean":
                print("bool")
                self.guardrail_boolean()
            elif self.current_type == "function":
                print("func")
                self.guardrail_function()

    def guardrail_string(self) -> None:
        self.mask_whitelist(None)
        self.mask_blacklist({'"}}'})

    def guardrail_number(self) -> None:
        self.mask_blacklist(None)
        self.mask_whitelist(Mask.MASK_NBR)
        valid_number = False
        number = self.response_text.split(f'"{self.current_key}":')[-1]
        if '.' in number:
            after_dot = number.split('.')[-1]
            if len(after_dot) > 0:
                valid_number = True
            if len(after_dot) == 3:
                self.mask_blacklist(None)
        if valid_number:
            schema, keylist, index = self.schema_stack[-1]
            if index >= len(keylist) - 1:
                print('\033[31m final close\033[0m')
                self.mask_whitelist(Mask.MASK_CLOSE_PARAM)
            else:
                print('\033[31m param close\033[0m')
                self.mask_whitelist(Mask.MASK_CLOSE_PARAM_NUM)

    def guardrail_boolean(self) -> None:
        self.mask_blacklist(None)
        self.mask_whitelist({'false', 'true'})
        if self.response_text.endswith(('true', 'fasle')):
            schema, keylist, index = self.schema_stack[-1]
            if index >= len(keylist) - 1:
                self.mask_whitelist(Mask.MASK_CLOSE_PARAM)
            else:
                self.mask_whitelist(Mask.MASK_CLOSE_PARAM_NUM)

    def guardrail_function(self) -> None:
        schema, keylist, index = self.schema_stack[-1]
        nested_schema: dict = schema[self.current_key].get('parameters', {})
        nested_keylist: list = list(nested_schema.keys())
        self.mask_blacklist(None)
        self.mask_whitelist({'":{"'})
        self.schema_stack[-1][2] += 1
        self.schema_stack.append([nested_schema, nested_keylist, 0])
        self.update_schema()
        self.state = State.EXPECTING_P_KEY

    # BLACKLIST MANAGER
    @singledispatchmethod
    def mask_blacklist(self, mask) -> None:
        """Default blacklist methode"""
        print("WARNING: Fell back to default mask_blacklist function [GuardRail > logit_sorter()]")

    @mask_blacklist.register(NoneType)
    def bl_from_null(self, mask) -> None:
        self.mask = np.full_like(self.logits, -float('inf'))

    @mask_blacklist.register(set)
    def bl_from_set(self, mask: set) -> None:
        for dict_index, dict_text in self.dictionary.items():
            if dict_text in mask:
                self.mask[dict_index] = -float('inf')

    @mask_blacklist.register(str)
    def bl_from_str(self, mask: str) -> None:
        mask_convert = mask.replace(' ', '\u0120').replace('\n', '\u010a')
        for dict_index, dict_text in self.dictionary.items():
            if dict_text in mask_convert:
                self.mask[dict_index] = -float('inf')

    # WHITELIST MANAGER
    @singledispatchmethod
    def mask_whitelist(self, mask) -> None:
        """Default whitelist methode"""
        print("WARNING: Fell back to default mask_whitelist function [GuardRail > logit_sorter()]")

    @mask_whitelist.register(type(None))
    def wl_from_null(self, mask: None) -> None:
        self.mask = np.full_like(self.logits, 0.0)

    @mask_whitelist.register(set)
    def wl_from_set(self, mask: set) -> None:
        for dict_index, dict_text in self.dictionary.items():
            if dict_text in mask:
                self.mask[dict_index] = 0.0

    @mask_whitelist.register(str)
    def wl_from_str(self, mask: str) -> None:
        # If inside prompt, respect user prompt pattern
        if self.state == State.INSIDE_PROMPT:
            self.update_response_text()
            current_response = self.response_text.split('":"')[-1]
            for dict_index, dict_text in self.dictionary.items():
                predicted_response = current_response + dict_text.replace('\u0120', ' ').replace('\u010a', '\n')
                if mask.startswith(predicted_response):
                    self.mask[dict_index] = 0.0
        else:
            mask_convert = mask.replace(' ', '\u0120').replace('\n', '\u010a')
            for dict_index, dict_text in self.dictionary.items():
                if dict_text in mask_convert:
                    self.mask[dict_index] = 0.0

    @mask_whitelist.register(list)
    def wl_from_list(self, mask: list[str]) -> None:
        self.update_response_text()
        current_response = self.response_text.split('":"')[-1]
        for dict_index, dict_text in self.dictionary.items():
            predicted_response = current_response + dict_text
            if any(name.startswith(predicted_response) for name in mask):
                self.mask[dict_index] = 0.0

    def update_response_text(self):
        self.response_text = "".join(self.dictionary[t] for t in self.response_ids)
        self.response_text = self.response_text.replace('\u0120', ' ').replace('\u010a', '\n')

    def clear_guardrail(self):
        self.response_text = ""
        self.function = ""
        self.schema_stack.clear()
        self.current_key = ""
        self.current_type = ""


class State(Enum):
    OPEN_JSON = 1
    EXPECTING_PROMPT = 2
    INSIDE_PROMPT = 3
    EXPECTING_NAME = 4
    INSIDE_FUNCTION = 5
    EXPECTING_PARAM = 6
    EXPECTING_P_KEY = 7
    INSIDE_PARAM = 8
    CLOSE_JSON = 9


class Mask():
    # \u0109 tab | \u0120 space | \u010a \n
    MASK_START = {'{"'}
    MASK_END = {'}'}
    MASK_CLOSEPRMPT = {'","', '"'}
    MASK_CLOSE_PARAM_KEY_NUM = {'":'}
    MASK_CLOSE_PARAM_NUM = {',"'}
    MASK_CLOSE_PARAM_KEY_STR = {'":"'}
    MASK_CLOSE_PARAM_STR = {'","'}
    MASK_CLOSE_PARAM = {'}}', '"}}', '}', '"}'}
    MASK_OPENSYM = {'{"'}
    MASK_BAN = {'"\u010a', '?"\u010a'}
    MASK_NBR = {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '.', '-'
    }
    MASK_CLOSENBR = {
        ',', '}'
    }
    MASK_BOOL = {
        'true', 'fasle'
    }
    MASK_ADD_KEY = {
        'a', 'b',
        '"', '":', ','
    }
    ASK_GREET_KEY = {
        's',
        '":"'
    }
    ASK_REV_KEY = {
        's',
        '"', '\u0120"', '":', ',\u0120', '\u0120'
    }
    MASK_ROOT_KEY = {
        'a',
        '"', '\u0120"', '":', ',\u0120', '\u0120'
    }
    MASK_REGEX_KEY = {
        'source_string', 'regex', 'replacement',
        '"', '\u0120"', '":', ',\u0120', '\u0120'
    }
    MASK_PROMPT_KEY = {'prompt', '":"'}
    MASK_NAME_KEY = {'"', 'name', '":"'}
    MASK_PARAM_KEY = {'"', 'parameters', '":{"'}
