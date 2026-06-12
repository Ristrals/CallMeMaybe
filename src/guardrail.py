#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   guardrail.py                                         :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: kmalfois <kmalfois@student.42.fr>            +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/05/27 11:49:53 by kmalfois            #+#    #+#            #
#   Updated: 2026/06/12 17:08:20 by kmalfois           ###   ########.fr      #
#                                                                             #
# ########################################################################### #


import numpy as np
import numpy.typing as npt
import time
import re
from types import NoneType
from enum import Enum
from llm_sdk import Small_LLM_Model as llm_model
from src.directive_regulator import DirectiveRegulator as dr
from src.lexicon import Lexicon as lex


class GuardRail():
    def __init__(
            self,
            model: llm_model,
            directive: dr,
            response_ids: list[int],
            lexicon: lex
    ) -> None:
        self.model: llm_model = model
        self.directive: dr = directive
        self.response_ids: list[int] = response_ids
        self.response_text: str = ""
        self.response_cursor: int = 0
        self.lexicon: lex = lexicon
        self.state: Enum = State.OPEN_JSON
        self.logits: npt.ArrayLike = []
        self.mask: npt.ArrayLike = np.zeros(len(self.lexicon.id_to_token), dtype=np.float32)
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
        self.mask.fill(-float('inf'))

        if self.state == State.OPEN_JSON:
            self.mask_blacklist(None)
            self.mask_whitelist(Mask.MASK_START)
        elif self.state == State.EXPECTING_PROMPT:
            self.mask_blacklist(None)
            self.mask_whitelist(Mask.MASK_PROMPT_KEY)
        elif self.state == State.INSIDE_PROMPT:
            self.mask_blacklist(None)
            self.mask_whitelist(self.directive.current_usr_prompt())
            if self.response_text.endswith(self.directive.current_usr_prompt()):
                self.mask_blacklist(None)
                self.mask_whitelist(Mask.MASK_CLOSEPRMPT)
        elif self.state == State.EXPECTING_NAME:
            self.mask_blacklist(None)
            self.mask_whitelist(Mask.MASK_NAME_KEY)
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
        elif self.state == State.EXPECTING_P_KEY:
            self.guardrail_parameter()
        elif self.state == State.INSIDE_PARAM:
            self.guardrail_parameter()
        elif self.state == State.CLOSE_JSON:
            self.mask_blacklist(None)
            self.mask_whitelist(Mask.MASK_END)

        self.logits = self.logits + self.mask
        next_token = int(np.argmax(self.logits))

        print(f"\033[33m{int(np.argmax(self.logits))} | "
              f"{self.lexicon.lex_decode([int(np.argmax(self.logits))])} | "
              f"{self.current_key}\033[0m")
        return next_token

    def update_state(self) -> None:
        self.update_response_text()

        # Removes unwanted spaces for an end of parameter descriptors
        norm_endline = re.sub(r',\s*"$', ',"', self.response_text)
        norm_endline = re.sub(r'\}\s*"$', '}"', norm_endline)

        if (self.state == State.OPEN_JSON and self.response_text.endswith('{"')):
            self.state = State.EXPECTING_PROMPT
        elif (self.state == State.EXPECTING_PROMPT and self.response_text.endswith('"prompt":"')):
            self.state = State.INSIDE_PROMPT
        elif (self.state == State.INSIDE_PROMPT and self.response_text.endswith('","')):
            self.state = State.EXPECTING_NAME
        elif (self.state == State.EXPECTING_NAME and self.response_text.endswith('"name":"')):
            self.state = State.INSIDE_FUNCTION
        elif (self.state == State.INSIDE_FUNCTION and self.response_text.endswith('","')):
            self.recover_func_name()
            self.state = State.EXPECTING_PARAM
        elif (self.state == State.EXPECTING_PARAM and self.response_text.endswith('"parameters":{"')):
            self.state = State.EXPECTING_P_KEY
        elif (self.state == State.EXPECTING_P_KEY and self.response_text.endswith('":{"')):
            schema, keylist, index = self.schema_stack[-1]
            nested_schema = schema[self.current_key].get('parameters', {})
            nested_keylist = list(nested_schema.keys())
            self.schema_stack[-1][2] += 1
            self.schema_stack.append([nested_schema, nested_keylist, 0])
            self.update_schema()
            self.state = State.EXPECTING_P_KEY
        elif (self.state == State.EXPECTING_P_KEY and (
            self.response_text.endswith('":"') or
            self.response_text.endswith('":')
        )):
            self.state = State.INSIDE_PARAM
        elif (self.state == State.EXPECTING_P_KEY and self.response_text.endswith('"}')):
            if len(self.schema_stack) > 1:
                self.schema_stack.pop()
                self.update_schema()
                schema, keylist, index = self.schema_stack[-1]
                if index < len(keylist):
                    self.state = State.EXPECTING_P_KEY
            else:
                self.state = State.CLOSE_JSON
        elif (self.state == State.INSIDE_PARAM and norm_endline.endswith(',"')):
            self.schema_stack[-1][2] += 1
            schema, keylist, index = self.schema_stack[-1]
            if len(self.schema_stack) > 1 and index >= len(keylist):
                self.schema_stack.pop()
                self.update_schema()
            else:
                self.update_schema()
            self.state = State.EXPECTING_P_KEY
        elif (self.state == State.INSIDE_PARAM and (
            norm_endline.endswith('"}') or
            norm_endline.endswith('}')
        )):
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
        func_name = self.response_text.split('"name":"')[-1].split('","')[0]
        for function in self.directive.func_data:
            if function['name'] == func_name:
                self.function = function['name']
                param_schema: dict = function.get('parameters', {})
                param_keylist: list = list(param_schema.keys())
                self.schema_stack.append([param_schema, param_keylist, 0])
                self.update_schema()
                break

    def update_schema(self) -> None:
        if self.schema_stack:
            schema, keylist, index = self.schema_stack[-1]
            if 0 <= index < len(keylist):
                self.current_key = keylist[index]
                self.current_type = schema[self.current_key]['type']
            else:
                self.current_key = ""
                self.current_type = ""

    # GUARDRAIL MAIN
    def guardrail_parameter(self) -> None:
        if self.state == State.EXPECTING_P_KEY:
            self.mask_blacklist(None)
            self.mask_whitelist(self.current_key)
            if self.response_text.endswith(self.current_key):
                if self.current_type == 'number' or self.current_type == 'boolean':
                    self.mask_whitelist(Mask.MASK_CLOSE_PARAM_KEY_NUM)
                elif self.current_type == 'string':
                    self.mask_whitelist(Mask.MASK_CLOSE_PARAM_KEY_STR)
                elif self.current_type == 'function':
                    self.mask_whitelist(('":{"',))
        elif self.state == State.INSIDE_PARAM:
            self.update_schema()
            print(f"\033[31m {self.current_key} | {self.current_type}\033[0m")
            if self.current_type == "string":
                print("\033[31m str gr\033[0m")
                self.guardrail_string()
            elif self.current_type == "number":
                print("\033[31m nbr gr\033[0m")
                self.guardrail_number()
            elif self.current_type == "boolean":
                print("\033[31m bool gr\033[0m")
                self.guardrail_boolean()
            elif self.current_type == "function":
                print("\033[31m func gr\033[0m")
                self.guardrail_function()

    # GUARDRAIL String
    def guardrail_string(self) -> None:
        self.mask_whitelist(None)
        self.mask_blacklist(('"}}', '"}}\u010a', '"},"', '","'))

    # GUARDRAIL Number
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
                self.mask_whitelist(Mask.MASK_CLOSE_PARAM)
            else:
                self.mask_whitelist(Mask.MASK_CLOSE_PARAM_NUM)

    # GUARDRAIL Boolean
    def guardrail_boolean(self) -> None:
        self.mask_blacklist(None)
        self.mask_whitelist(Mask.MASK_BOOL)
        if self.response_text.endswith(('true', 'fasle')):
            schema, keylist, index = self.schema_stack[-1]
            if index >= len(keylist) - 1:
                self.mask_whitelist(Mask.MASK_CLOSE_PARAM)
            else:
                self.mask_whitelist(Mask.MASK_CLOSE_PARAM_NUM)

    # GUARDRAIL Functions
    def guardrail_function(self) -> None:
        schema, keylist, index = self.schema_stack[-1]
        nested_schema: dict = schema[self.current_key].get('parameters', {})
        nested_keylist: list = list(nested_schema.keys())
        self.mask_blacklist(None)
        self.mask_whitelist(('":{"',))
        self.schema_stack[-1][2] += 1
        self.schema_stack.append([nested_schema, nested_keylist, 0])
        self.update_schema()
        self.state = State.EXPECTING_P_KEY

    # GUARDRAIL Clear
    def guardrail_clear(self):
        self.response_text = ""
        self.function = ""
        self.schema_stack.clear()
        self.current_key = ""
        self.current_type = ""
        self.response_cursor = 0

    # LIST MANAGER
    def list_manager(self, mask: tuple | list | str, mask_value: float) -> None:
        self.update_response_text()
        if isinstance(mask, list):
            current_fragment = self.response_text.split('":"')[-1] if self.response_text else ""
            for dict_index, dict_text in self.lexicon.id_to_token.items():
                predicted_response = current_fragment + dict_text
                if any(name.startswith(predicted_response) for name in mask):
                    self.mask[dict_index] = mask_value

        elif isinstance(mask, tuple):
            for item in mask:
                token_ids = self.lexicon.lex_encode(str(item))
                for token_id in token_ids:
                    self.mask[token_id] = mask_value

        else:
            if self.state == State.INSIDE_PROMPT:
                current_fragment = self.response_text.split('":"')[-1] if self.response_text else ""
                if mask.startswith(current_fragment):
                    remaining_prompt = mask[len(current_fragment):]
                    token_ids = self.lexicon.lex_encode(remaining_prompt)
                    if token_ids:
                        self.mask[token_ids[0]] = mask_value
            else:
                token_ids = self.lexicon.lex_encode(mask)
                for token_id in token_ids:
                    self.mask[token_id] = mask_value

    # BLACKLIST MANAGER
    @singledispatchmethod
    def mask_blacklist(self, mask) -> None:
        """Default blacklist methode"""
        print("WARNING: Fell back to default mask_blacklist function [GuardRail > logit_sorter()]")

    @mask_blacklist.register(NoneType)
    def bl_from_null(self, mask) -> None:
        self.mask = np.full_like(self.logits, -float('inf'))

    @mask_blacklist.register(tuple)
    @mask_blacklist.register(list)
    @mask_blacklist.register(str)
    def bl_global(self, mask: tuple | list | str) -> None:
        self.list_manager(mask, -float('inf'))

    # WHITELIST MANAGER
    @singledispatchmethod
    def mask_whitelist(self, mask) -> None:
        """Default whitelist methode"""
        print("WARNING: Fell back to default mask_whitelist function [GuardRail > logit_sorter()]")

    @mask_whitelist.register(type(None))
    def wl_from_null(self, mask: None) -> None:
        self.mask = np.full_like(self.logits, 0.0)

    @mask_whitelist.register(tuple)
    @mask_whitelist.register(list)
    @mask_whitelist.register(str)
    def wl_global(self, mask: tuple) -> None:
        self.list_manager(mask, 0.0)

    def update_response_text(self):
        if self.response_cursor < len(self.response_ids):
            new_ids = self.response_ids[self.response_cursor:]
            raw_text = "".join(self.lexicon.id_to_token[i] for i in new_ids if i in self.lexicon.id_to_token)
            clear_text = raw_text.replace('\u0120', ' ').replace('\u010a', '\n')
            self.response_text += clear_text
            self.response_cursor = len(self.response_ids)

    # CHRONO WRAPPER
    def profile_section(self, name: str, start_time: float) -> float:
        """Prints the elapsed time for a section and returns the new baseline time."""
        # t = time.perf_counter()
        # t = self.profile_section("Argmax Token Selection", t)
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        print(f"   [TIMER] {name}: {duration_ms:.4f} ms")
        return end_time


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
    MASK_START = ('{"',)
    MASK_END = ('}',)
    MASK_CLOSEPRMPT = ('","', '"')
    MASK_CLOSE_PARAM_KEY_NUM = ('":',)
    MASK_CLOSE_PARAM_NUM = (',"',)
    MASK_CLOSE_PARAM_KEY_STR = ('":"',)
    MASK_CLOSE_PARAM_STR = ('","',)
    MASK_CLOSE_PARAM = ('}}', '"}}', '}', '"}')
    MASK_OPENSYM = ('{"',)
    MASK_NBR = (
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '.', '-'
    )
    MASK_BOOL = (
        'true', 'fasle'
    )
    MASK_PROMPT_KEY = ('prompt', '":"')
    MASK_NAME_KEY = ('"', 'name', '":"')
    MASK_PARAM_KEY = ('"', 'parameters', '":{"')
