import os
import shutil
import uuid
import numpy as np
import tensorflow as tf
import math

LIB_NAME = 'segment_median'


def load_op_module(lib_name):
  """
  Load TensorFlow operator library.
  """
  # use absolute path so that ops.py can be called from other directory
  lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cpp/build/lib{0}.so'.format(lib_name))
  # duplicate library with a random new name so that
  # a running program will not be interrupted when the original library is updated
  lib_copy_path = '/tmp/lib{0}_{1}.so'.format(str(uuid.uuid4())[:8], LIB_NAME)
  shutil.copyfile(lib_path, lib_copy_path)
  oplib = tf.load_op_library(lib_copy_path)
  return oplib

op_module = load_op_module(LIB_NAME)

class SegmentMedianTest(tf.test.TestCase):
  def testSegmentMedian(self):
    # map C++ operators to python objects
    segment_median = op_module.segment_median
    with self.test_session():
      result = segment_median([[i for i in range(1)],[i*4 for i in range(1)],[i*3 for i in range(1)],[i*2 for i in range(1)]], [1,1,1,1])
      print(result.eval())
      result = segment_median([[i for i in range(2)],[i*4 for i in range(2)],[i*3 for i in range(2)],[i*2 for i in range(2)]], [1,1,1,1])
      print(result.eval())
      result = segment_median([[i for i in range(10000)],[i*4 for i in range(10000)],[i*3 for i in range(10000)],[i*2 for i in range(10000)]], [1,1,2,3])
      print(result.eval())
      result = segment_median([[i for i in range(10000)]], [1])
      print(result.eval())


if __name__ == "__main__":
  tf.test.main()
