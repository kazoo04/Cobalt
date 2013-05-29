#!/usr/bin/env python
#coding utf-8

import os
import sys
import re
import struct
import numpy
import scipy.cluster
from random import Random

num_files = 10

def open_feature_file(filename):
  v = []
  vectors = []

  with open(filename, 'rb') as f:
    for vals in iter(lambda: f.read(64), b''):
      for val in vals:
        v.append(struct.unpack('<B', val)[0])

      vectors.append(v)
      v = []

  return vectors


if len(sys.argv) != 2:
  print 'usage'
  print '\tpython ' + sys.argv[0] + ' dir'
  sys.exit(1)

files = os.listdir(sys.argv[1])

rand = Random()
rand.shuffle(files)
files = files[:num_files]

vectors = []

for f in files:
  if not re.match(r'.+\.feature', f, re.IGNORECASE):
    continue

  vectors.extend(open_feature_file(sys.argv[1] + '/' + f))

X = numpy.array(vectors)

codebook, destortion = scipy.cluster.vq.kmeans2(X, 1e2, iter=2, thresh=1e-3)
for row in codebook:
  print ','.join([str(r) for r in row])

