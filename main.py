#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   main.py                                              :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: kmalfois <kmalfois@student.42.fr>            +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/05/20 13:34:08 by kmalfois            #+#    #+#            #
#   Updated: 2026/05/28 15:09:58 by kmalfois           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import sys
import json
from pydantic import BaseModel
from llm_sdk import Small_LLM_Model
from src.crucible import Crucible as cr


if __name__ == "__main__":
    with open(sys.argv[1], "r") as func_file:
        func = json.load(func_file)
    with open(sys.argv[2], "r") as prmpt_file:
        prmpt = json.load(prmpt_file)

    crucible = cr(prmpt, func)
    crucible.response_gen()
    print(crucible)

    # sample_id = [666, 777, 42, 23678]
    # for id in sample_id:
    #     true_word = tokenizer.decode(id)
    #     print(f"{id}: {true_word}")
    # sample_str = '}\n'
    # result = model.encode(sample_str).tolist()[0]
    # print(result)
    # with open(gwen_dict, "r", encoding="utf-8") as dictionary:
    #     vocabulary = json.load(dictionary)
    # vocab_map = {int(v): k for k, v in vocabulary.items()}
    # print(f"'{vocab_map[532]}'")
