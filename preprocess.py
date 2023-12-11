import os
import sys
import json
import csv
import glob

import re

with open("data2.csv", "w") as c:
    writer = csv.writer(c)
    writer.writerow(["content"] + ["bias"])
    for i, fname in enumerate(glob.glob("data/*")):
        with open(fname) as fin:
            print(i)
            if (i == 5000):
                break
            row = json.load(fin)
            content = row["content"];
            # print(content, end='\n\n')
            content = re.sub(r"[^a-zA-Z\s]+", "", content)
            content = re.sub(r"\s+", " ", content)
            # print(content)
            writer.writerow([content] + [row["bias_text"]])
