#!/usr/bin/env python
#coding: utf-8

import os
import sys
import re
import subprocess

if len(sys.argv) <= 2:
  print 'usage:'
  print '\tpython ' + sys.argv[0] + ' src_dir dist_dir'
  sys.exit(1)

files = os.listdir(sys.argv[1])

for f in files:
  match = re.match(r'(.+)\.(jpg|png)', f, re.IGNORECASE)
  if match:
    img_filename = sys.argv[1] + '/' + f
    feature_filename = sys.argv[2] + '/' + match.group(1) + '.feature'
    cmd = './extract_descriptor ' + img_filename + ' > ' + feature_filename
    subprocess.call(cmd, shell=True)

print 'done.'
