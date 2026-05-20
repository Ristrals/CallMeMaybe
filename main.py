#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   main.py                                              :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: kmalfois <kmalfois@student.42.fr>            +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/05/20 13:34:08 by kmalfois            #+#    #+#            #
#   Updated: 2026/05/20 20:37:33 by kmalfois           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
import numpy
from pydantic import BaseModel
from llm_sdk import Small_LLM_Model

if __name__ == "__main__":
    model = Small_LLM_Model()
    gwen_dict = model.get_path_to_vocab_file()
    with open(gwen_dict, "r", encoding="utf-8") as dictionary:
        vocabulary = json.load(dictionary)
    vocab_map = {int(v): k for k, v in vocabulary.items()}
    print(vocab_map)
