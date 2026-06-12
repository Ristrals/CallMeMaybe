#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   lexicon.py                                           :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: kmalfois <kmalfois@student.42.fr>            +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/10 12:56:05 by kmalfois            #+#    #+#            #
#   Updated: 2026/06/12 17:08:21 by kmalfois           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
from functools import lru_cache
from llm_sdk import Small_LLM_Model as llm_model


class Lexicon():
    def __init__(self, llm_model: llm_model):
        self.model = llm_model
        self.build_lexicon()

    def build_lexicon(self) -> None:
        llm_dict = self.model.get_path_to_vocab_file()
        # llm_dict = self.model.llm.get_path_to_vocab_file()
        with open(llm_dict, "r", encoding="utf-8") as vocab_file:
            vocabulary = json.load(vocab_file)
        self._token_to_id: dict[str, int] = vocabulary
        id_to_token = {int(v): k for k, v in vocabulary.items()}
        self._id_to_token: dict[int, str] = id_to_token

    @property
    def token_to_id(self) -> dict[str, int]:
        return self._token_to_id

    @property
    def id_to_token(self) -> dict[str, int]:
        return self._id_to_token

    # CUSTOM DECODE | ENCODE
    def lex_decode(self, token_ids: list[int]) -> str:
        raw_text = [self.id_to_token[id] for
                    id in token_ids if id in self.id_to_token]
        clear_text = "".join(raw_text).replace('\u0120', ' ').replace('\n', '\u010a')
        return clear_text

    @lru_cache(maxsize=None)
    def lex_encode(self, clear_text: str) -> list[int]:
        encoded_text = clear_text.replace(' ', '\u0120').replace('\n', '\u010a')
        token_ids = []
        i = 0
        ct_length = len(encoded_text)
        while i < ct_length:
            match = False
            for j in range(ct_length, i, -1):
                substring = encoded_text[i:j]
                if substring in self.token_to_id:
                    token_ids.append(self.token_to_id[substring])
                    i = j
                    match = True
                    break
            if not match:
                i += 1
        return token_ids
