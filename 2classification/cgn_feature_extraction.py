# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import re
import os
import xml.etree.ElementTree as ET

# <codecell>

readPath = "train/"

# <codecell>

for filename in os.listdir(readPath):
    print ('Working on file:', filename
    f = open(readPath + filename,'r')
    #sections = re.findall(r'<section.*?id=.*?>.*?</section>', fileAsString, re.DOTALL)
    
    #Create ElementTree representing the XML structure of the laws
    #tree = ET.parse(readPath + filename)
    #root = tree.getroot()

# <codecell>


