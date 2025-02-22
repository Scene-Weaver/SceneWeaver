import json
import re
from functools import reduce

import prompts as prompts
from gpt import GPT4 as gpt
import os




gpt = gpt()

results = dict()

basedir = "/mnt/fillipo/yandan/metascene/export_stage2_sm/scene0001_00/0/"

candidates_fpaths = []
for file in os.listdir(basedir):
    candidates_fpaths.append(f"{basedir}/{file}")

prompt_payload = gpt.payload_front_pose("sofa",candidates_fpaths)
gpt_text_response = gpt(payload=prompt_payload, verbose=True)
print(gpt_text_response)

