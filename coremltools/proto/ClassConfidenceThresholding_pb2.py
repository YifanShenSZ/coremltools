# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ClassConfidenceThresholding.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import DataStructures_pb2 as DataStructures__pb2
try:
  FeatureTypes__pb2 = DataStructures__pb2.FeatureTypes__pb2
except AttributeError:
  FeatureTypes__pb2 = DataStructures__pb2.FeatureTypes_pb2

from .DataStructures_pb2 import *

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n!ClassConfidenceThresholding.proto\x12\x14\x43oreML.Specification\x1a\x14\x44\x61taStructures.proto\"h\n\x1b\x43lassConfidenceThresholding\x12I\n\x15precisionRecallCurves\x18\x64 \x03(\x0b\x32*.CoreML.Specification.PrecisionRecallCurveB\x02H\x03P\x00\x62\x06proto3')



_CLASSCONFIDENCETHRESHOLDING = DESCRIPTOR.message_types_by_name['ClassConfidenceThresholding']
ClassConfidenceThresholding = _reflection.GeneratedProtocolMessageType('ClassConfidenceThresholding', (_message.Message,), {
  'DESCRIPTOR' : _CLASSCONFIDENCETHRESHOLDING,
  '__module__' : 'ClassConfidenceThresholding_pb2'
  # @@protoc_insertion_point(class_scope:CoreML.Specification.ClassConfidenceThresholding)
  })
_sym_db.RegisterMessage(ClassConfidenceThresholding)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'H\003'
  _CLASSCONFIDENCETHRESHOLDING._serialized_start=81
  _CLASSCONFIDENCETHRESHOLDING._serialized_end=185
# @@protoc_insertion_point(module_scope)