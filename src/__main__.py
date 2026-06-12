#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   __main__.py                                          :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: kmalfois <kmalfois@student.42.fr>            +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/10 12:38:43 by kmalfois            #+#    #+#            #
#   Updated: 2026/06/12 11:09:31 by kmalfois           ###   ########.fr      #
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
