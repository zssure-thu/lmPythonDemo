# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 16:02:37 2017

@author: Administrator
"""

import json
import numpy as np
f = open("d:\\circle_matrix.json",encoding='utf-8')
settings = json.load(f)
out = []
i = 0
for setting in settings:
    out = []
    for (d,x) in setting.items():
    #if int(d)%4 == 3:
        out.append(x)
    print(out)

f = open("d:\\circle_floodfill.json",encoding='utf-8')
settings = json.load(f)
out = []
i = 0
for setting in settings:
    out = []
    for (d,x) in setting.items():
    #if int(d)%4 == 3:
        out.append(x)
    print(out)