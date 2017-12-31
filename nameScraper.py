#!/usr/local/bin/python3

from bs4 import BeautifulSoup
from lxml import html
import json
import requests
import time
import pdb

# time.sleep(1)
ethnicityList = ["African", "Arabic", "Chinese", "Danish", "English", "Germanic", "Greek", "Indian", "Japanese", "Korean", "Russian"]

f = open('names.txt','w')
for ethnicity in ethnicityList:
    for i in range(2,100):

        # Query website
        time.sleep(1)
        ethnicity = ethnicity.lower()
        url = "INSERT_URL_HERE"
        page = requests.get(url)

        # Ignore if out of range
        # if i == 15:
        #     pdb.set_trace()
        if page.status_code==404:
            print("Broke at %s %d" % (ethnicity, i))
            break
        # tree = html.fromstring(page.content)

        # Convert to html legible/parsible format
        soup = BeautifulSoup(page.content, 'html.parser')
        s=soup.find('script', type='text/javascript')

        # Extract list of names on this page
        for element in s.next_elements:
          if "var names" in element:
            nameList = element

        # Convert names from json dictionary to python dictionary
        names = nameList.splitlines()[4]
        nameDict = json.loads(names[14:-1])
        for d in nameDict:
            print(d['name']+","+ethnicity)
            f.write(d['name']+","+ethnicity+"\n")
        if not(nameDict):
            break

f.close()
