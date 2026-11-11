#include "model_profile.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "platform_config.h"
#include "public_testbench.h"

void reset_profile_state();
void print_profile_summary();

namespace {

static int total_pass_count = 0;
static int total_fail_count = 0;
static int global_input_counter = 0;

struct BertTestSample {
  const char* name;
  const int32_t* input_ids;
  const int32_t* attention_mask;
  const float* golden_outputs;
};

const BertTestSample kSamples[] = {
    {"public_test_0", public_input_ids[0], public_attention_mask[0], public_golden_output[0]},
    {"public_test_1", public_input_ids[1], public_attention_mask[1], public_golden_output[1]},
    {"public_test_2", public_input_ids[2], public_attention_mask[2], public_golden_output[2]},
    {"public_test_3", public_input_ids[3], public_attention_mask[3], public_golden_output[3]},
    {"public_test_4", public_input_ids[4], public_attention_mask[4], public_golden_output[4]},
};

const size_t kSampleCount = sizeof(kSamples) / sizeof(kSamples[0]);
const size_t kBertSequenceLength = 125;
const size_t kBertLogitsCount = 2;

void clear_input(TfLiteTensor* input) {
  memset(input->data.raw, 0, input->bytes);
}

void bert_prepare_input(TfLiteTensor* input, size_t sample_index) {
  if (input == nullptr || sample_index >= kSampleCount) {
    return;
  }

  const BertTestSample& sample = kSamples[sample_index];

  if (input->type != kTfLiteInt32) {
    printf("Input %s: unsupported tensor type=%d; expected kTfLiteInt32.\n", sample.name, input->type);
    clear_input(input);
    return;
  }

  const size_t expected_bytes = kBertSequenceLength * sizeof(int32_t);
  if (input->bytes != expected_bytes) {
    clear_input(input);
    return;
  }

  if (input->name != nullptr && strstr(input->name, "input_ids") != nullptr) {
    printf("Loading input ids for %s...\n", sample.name);
    memcpy(input->data.i32, sample.input_ids, expected_bytes);
  } 
  else if (input->name != nullptr && strstr(input->name, "attention_mask") != nullptr) {
    printf("Loading attention mask for %s...\n", sample.name);
    memcpy(input->data.i32, sample.attention_mask, expected_bytes);
  } 
  else {
    if (global_input_counter % 2 == 0) {
      printf("Loading attention mask for %s...\n", sample.name);
      memcpy(input->data.i32, sample.attention_mask, expected_bytes);
    } else {
      printf("Loading input IDs for %s...\n", sample.name);
      memcpy(input->data.i32, sample.input_ids, expected_bytes);
    }
    global_input_counter++;
  }
}

void bert_verify_output(const TfLiteTensor* output, size_t sample_index) {
  if (output == nullptr || sample_index >= kSampleCount) {
    return;
  }

  const BertTestSample& sample = kSamples[sample_index];

  if (output->type != kTfLiteFloat32) {
    printf("Output %s: golden data requires float32. [FAIL]\n", sample.name);
    total_fail_count++;
    return;
  }

  const size_t output_elements = output->bytes / sizeof(float);
  if (output_elements != kBertLogitsCount) { 
    printf("Output %s: tensor=%u elements, expected=%u. [FAIL]\n", sample.name,
           (unsigned)output_elements, (unsigned)kBertLogitsCount);
    total_fail_count++;
    return;
  }

  int error_count = 0;
  const float epsilon = 1e-4f; 

  for (size_t i = 0; i < output_elements; ++i) {
    float actual_val;
    memcpy(&actual_val, (uint8_t*)output->data.data + i * sizeof(float), sizeof(float));

    float diff = fabsf(actual_val - sample.golden_outputs[i]);
    bool matches = (diff <= epsilon);
    
    printf("Output[%u]: Actual=%.6f, Golden=%.6f [%s]\n", 
           (unsigned)i, actual_val, sample.golden_outputs[i], matches ? "PASS" : "FAIL");
           
    if (!matches) {
      ++error_count;
    }
  }

  if (error_count == 0) {
    total_pass_count++;
    printf("[Status] %s: ALL OUTPUTS PASS.\n\n", sample.name);
  } else {
    total_fail_count++;
    printf("[Status] %s: FAIL with %d errors.\n\n", sample.name, error_count);
  }
}

}  // namespace

void reset_profile_state() {
  total_pass_count = 0;
  total_fail_count = 0;
  global_input_counter = 0;
}

void print_profile_summary() {
  printf("\n========== Summary ==========\n");
  printf("Total PASS: %d\n", total_pass_count);
  printf("Total FAIL: %d\n", total_fail_count);
}

const ModelProfile* model_profile_get(void) {
  static const ModelProfile kProfile = {
      "bert_medium_classifier",
      "public text fixtures",
      "golden outputs check",
      bert_prepare_input,
      bert_verify_output,
      0, 
      0,
      0,
      0,
      MODEL_TOLERANCE,
      5,
  };
  return &kProfile;
}