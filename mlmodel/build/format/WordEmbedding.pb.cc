// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: WordEmbedding.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "WordEmbedding.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
// @@protoc_insertion_point(includes)

namespace CoreML {
namespace Specification {
namespace CoreMLModels {
class WordEmbeddingDefaultTypeInternal : public ::google::protobuf::internal::ExplicitlyConstructed<WordEmbedding> {
} _WordEmbedding_default_instance_;

namespace protobuf_WordEmbedding_2eproto {

PROTOBUF_CONSTEXPR_VAR ::google::protobuf::internal::ParseTableField
    const TableStruct::entries[] = {
  {0, 0, 0, ::google::protobuf::internal::kInvalidMask, 0, 0},
};

PROTOBUF_CONSTEXPR_VAR ::google::protobuf::internal::AuxillaryParseTableField
    const TableStruct::aux[] = {
  ::google::protobuf::internal::AuxillaryParseTableField(),
};
PROTOBUF_CONSTEXPR_VAR ::google::protobuf::internal::ParseTable const
    TableStruct::schema[] = {
  { NULL, NULL, 0, -1, -1, false },
};


void TableStruct::Shutdown() {
  _WordEmbedding_default_instance_.Shutdown();
}

void TableStruct::InitDefaultsImpl() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::google::protobuf::internal::InitProtobufDefaults();
  ::CoreML::Specification::protobuf_DataStructures_2eproto::InitDefaults();
  _WordEmbedding_default_instance_.DefaultConstruct();
}

void InitDefaults() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &TableStruct::InitDefaultsImpl);
}
void AddDescriptorsImpl() {
  InitDefaults();
  ::CoreML::Specification::protobuf_DataStructures_2eproto::AddDescriptors();
  ::google::protobuf::internal::OnShutdown(&TableStruct::Shutdown);
}

void AddDescriptors() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &AddDescriptorsImpl);
}
#ifdef GOOGLE_PROTOBUF_NO_STATIC_INITIALIZER
// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;
#endif  // GOOGLE_PROTOBUF_NO_STATIC_INITIALIZER

}  // namespace protobuf_WordEmbedding_2eproto


// ===================================================================

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int WordEmbedding::kRevisionFieldNumber;
const int WordEmbedding::kLanguageFieldNumber;
const int WordEmbedding::kModelParameterDataFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

WordEmbedding::WordEmbedding()
  : ::google::protobuf::MessageLite(), _internal_metadata_(NULL) {
  if (GOOGLE_PREDICT_TRUE(this != internal_default_instance())) {
    protobuf_WordEmbedding_2eproto::InitDefaults();
  }
  SharedCtor();
  // @@protoc_insertion_point(constructor:CoreML.Specification.CoreMLModels.WordEmbedding)
}
WordEmbedding::WordEmbedding(const WordEmbedding& from)
  : ::google::protobuf::MessageLite(),
      _internal_metadata_(NULL),
      _cached_size_(0) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  language_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.language().size() > 0) {
    language_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.language_);
  }
  modelparameterdata_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.modelparameterdata().size() > 0) {
    modelparameterdata_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.modelparameterdata_);
  }
  revision_ = from.revision_;
  // @@protoc_insertion_point(copy_constructor:CoreML.Specification.CoreMLModels.WordEmbedding)
}

void WordEmbedding::SharedCtor() {
  language_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  modelparameterdata_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  revision_ = 0u;
  _cached_size_ = 0;
}

WordEmbedding::~WordEmbedding() {
  // @@protoc_insertion_point(destructor:CoreML.Specification.CoreMLModels.WordEmbedding)
  SharedDtor();
}

void WordEmbedding::SharedDtor() {
  language_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  modelparameterdata_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}

void WordEmbedding::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const WordEmbedding& WordEmbedding::default_instance() {
  protobuf_WordEmbedding_2eproto::InitDefaults();
  return *internal_default_instance();
}

WordEmbedding* WordEmbedding::New(::google::protobuf::Arena* arena) const {
  WordEmbedding* n = new WordEmbedding;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void WordEmbedding::Clear() {
// @@protoc_insertion_point(message_clear_start:CoreML.Specification.CoreMLModels.WordEmbedding)
  language_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  modelparameterdata_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  revision_ = 0u;
}

bool WordEmbedding::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:CoreML.Specification.CoreMLModels.WordEmbedding)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(16383u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // uint32 revision = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(8u)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 input, &revision_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string language = 10;
      case 10: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(82u)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_language()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->language().data(), this->language().length(),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "CoreML.Specification.CoreMLModels.WordEmbedding.language"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // bytes modelParameterData = 100;
      case 100: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(802u)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadBytes(
                input, this->mutable_modelparameterdata()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormatLite::SkipField(input, tag));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:CoreML.Specification.CoreMLModels.WordEmbedding)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:CoreML.Specification.CoreMLModels.WordEmbedding)
  return false;
#undef DO_
}

void WordEmbedding::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:CoreML.Specification.CoreMLModels.WordEmbedding)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // uint32 revision = 1;
  if (this->revision() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt32(1, this->revision(), output);
  }

  // string language = 10;
  if (this->language().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->language().data(), this->language().length(),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "CoreML.Specification.CoreMLModels.WordEmbedding.language");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      10, this->language(), output);
  }

  // bytes modelParameterData = 100;
  if (this->modelparameterdata().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteBytesMaybeAliased(
      100, this->modelparameterdata(), output);
  }

  // @@protoc_insertion_point(serialize_end:CoreML.Specification.CoreMLModels.WordEmbedding)
}

size_t WordEmbedding::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:CoreML.Specification.CoreMLModels.WordEmbedding)
  size_t total_size = 0;

  // string language = 10;
  if (this->language().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->language());
  }

  // bytes modelParameterData = 100;
  if (this->modelparameterdata().size() > 0) {
    total_size += 2 +
      ::google::protobuf::internal::WireFormatLite::BytesSize(
        this->modelparameterdata());
  }

  // uint32 revision = 1;
  if (this->revision() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::UInt32Size(
        this->revision());
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void WordEmbedding::CheckTypeAndMergeFrom(
    const ::google::protobuf::MessageLite& from) {
  MergeFrom(*::google::protobuf::down_cast<const WordEmbedding*>(&from));
}

void WordEmbedding::MergeFrom(const WordEmbedding& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:CoreML.Specification.CoreMLModels.WordEmbedding)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.language().size() > 0) {

    language_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.language_);
  }
  if (from.modelparameterdata().size() > 0) {

    modelparameterdata_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.modelparameterdata_);
  }
  if (from.revision() != 0) {
    set_revision(from.revision());
  }
}

void WordEmbedding::CopyFrom(const WordEmbedding& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:CoreML.Specification.CoreMLModels.WordEmbedding)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool WordEmbedding::IsInitialized() const {
  return true;
}

void WordEmbedding::Swap(WordEmbedding* other) {
  if (other == this) return;
  InternalSwap(other);
}
void WordEmbedding::InternalSwap(WordEmbedding* other) {
  language_.Swap(&other->language_);
  modelparameterdata_.Swap(&other->modelparameterdata_);
  std::swap(revision_, other->revision_);
  std::swap(_cached_size_, other->_cached_size_);
}

::std::string WordEmbedding::GetTypeName() const {
  return "CoreML.Specification.CoreMLModels.WordEmbedding";
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// WordEmbedding

// uint32 revision = 1;
void WordEmbedding::clear_revision() {
  revision_ = 0u;
}
::google::protobuf::uint32 WordEmbedding::revision() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.CoreMLModels.WordEmbedding.revision)
  return revision_;
}
void WordEmbedding::set_revision(::google::protobuf::uint32 value) {
  
  revision_ = value;
  // @@protoc_insertion_point(field_set:CoreML.Specification.CoreMLModels.WordEmbedding.revision)
}

// string language = 10;
void WordEmbedding::clear_language() {
  language_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
const ::std::string& WordEmbedding::language() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.CoreMLModels.WordEmbedding.language)
  return language_.GetNoArena();
}
void WordEmbedding::set_language(const ::std::string& value) {
  
  language_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:CoreML.Specification.CoreMLModels.WordEmbedding.language)
}
#if LANG_CXX11
void WordEmbedding::set_language(::std::string&& value) {
  
  language_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:CoreML.Specification.CoreMLModels.WordEmbedding.language)
}
#endif
void WordEmbedding::set_language(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  language_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:CoreML.Specification.CoreMLModels.WordEmbedding.language)
}
void WordEmbedding::set_language(const char* value, size_t size) {
  
  language_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:CoreML.Specification.CoreMLModels.WordEmbedding.language)
}
::std::string* WordEmbedding::mutable_language() {
  
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.CoreMLModels.WordEmbedding.language)
  return language_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
::std::string* WordEmbedding::release_language() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.CoreMLModels.WordEmbedding.language)
  
  return language_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
void WordEmbedding::set_allocated_language(::std::string* language) {
  if (language != NULL) {
    
  } else {
    
  }
  language_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), language);
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.CoreMLModels.WordEmbedding.language)
}

// bytes modelParameterData = 100;
void WordEmbedding::clear_modelparameterdata() {
  modelparameterdata_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
const ::std::string& WordEmbedding::modelparameterdata() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.CoreMLModels.WordEmbedding.modelParameterData)
  return modelparameterdata_.GetNoArena();
}
void WordEmbedding::set_modelparameterdata(const ::std::string& value) {
  
  modelparameterdata_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:CoreML.Specification.CoreMLModels.WordEmbedding.modelParameterData)
}
#if LANG_CXX11
void WordEmbedding::set_modelparameterdata(::std::string&& value) {
  
  modelparameterdata_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:CoreML.Specification.CoreMLModels.WordEmbedding.modelParameterData)
}
#endif
void WordEmbedding::set_modelparameterdata(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  modelparameterdata_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:CoreML.Specification.CoreMLModels.WordEmbedding.modelParameterData)
}
void WordEmbedding::set_modelparameterdata(const void* value, size_t size) {
  
  modelparameterdata_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:CoreML.Specification.CoreMLModels.WordEmbedding.modelParameterData)
}
::std::string* WordEmbedding::mutable_modelparameterdata() {
  
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.CoreMLModels.WordEmbedding.modelParameterData)
  return modelparameterdata_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
::std::string* WordEmbedding::release_modelparameterdata() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.CoreMLModels.WordEmbedding.modelParameterData)
  
  return modelparameterdata_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
void WordEmbedding::set_allocated_modelparameterdata(::std::string* modelparameterdata) {
  if (modelparameterdata != NULL) {
    
  } else {
    
  }
  modelparameterdata_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), modelparameterdata);
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.CoreMLModels.WordEmbedding.modelParameterData)
}

#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace CoreMLModels
}  // namespace Specification
}  // namespace CoreML

// @@protoc_insertion_point(global_scope)