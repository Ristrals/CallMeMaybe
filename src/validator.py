#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   validator.py                                         :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: kmalfois <kmalfois@student.42.fr>            +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/05/20 11:31:47 by kmalfois            #+#    #+#            #
#   Updated: 2026/05/21 15:33:14 by kmalfois           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from pydantic import BaseModel


class JSONValidator(BaseModel):
    pass
