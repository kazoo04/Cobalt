#!/usr/bin/env python
#coding utf-8

import os
import sys
import re
import struct
import numpy
import scipy.cluster
import random

num_dimentions = 64
num_files = 4096
num_features_per_file = 1024

def open_feature_file(filename):
  vectors = []

  with open(filename, 'rb') as f:
    for vals in iter(lambda: f.read(num_dimentions), b''):
      vectors.append([struct.unpack('<B', x)[0] for x in vals])

  return random.sample(vectors, num_features_per_file)

if len(sys.argv) != 2:
  print 'usage'
  print '\tpython ' + sys.argv[0] + ' dir'
  sys.exit(1)

files = os.listdir(sys.argv[1])

random.shuffle(files)
files = files[:num_files]

vectors = []

for f in files:
  if not re.match(r'.+\.feature', f, re.IGNORECASE):
    continue

  vectors.extend(open_feature_file(sys.argv[1] + '/' + f))

X = numpy.array(vectors)

codebook, destortion = scipy.cluster.vq.kmeans2(X, 1e5, iter=5, thresh=1e-8)
for row in codebook:
  print ','.join([str(r) for r in row])

