#include "model_io.h"

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "model_profile.h"
#include "platform_config.h"

namespace {

int abs_int(int value) {
  return value < 0 ? -value : value;
}

const ModelProfile* active_profile(void) {
  static const ModelProfile kFallbackProfile = {
      "fallback",
      "zeros",
      "skip",
      0,
      0,
      0,
      0,
      0,
      0,
      MODEL_TOLERANCE, // 現為 float，可直接吃 1e-4 巨集
      1,
  };
  const ModelProfile* profile = model_profile_get();
  return profile == 0 ? &kFallbackProfile : profile;
}

}  // namespace

void model_io_print_profile(void) {
  const ModelProfile* profile = active_profile();
  printf("Profile: %s\n", profile->name);
  printf("Input profile: %s\n", profile->input_name);
  
  // 處理裸機 printf 不支援 float 的問題，將 float 拆成整數與小數部分列印
  int tol_int = (int)profile->tolerance;
  int tol_frac = (int)((profile->tolerance - tol_int) * 1000000.0f);
  
  printf("Output profile: %s, tolerance=%d.%06d\n", profile->output_name,
         tol_int, tol_frac);
  printf("Samples: %u\n", (unsigned)model_io_sample_count());
}

size_t model_io_sample_count(void) {
  const ModelProfile* profile = active_profile();
  return profile->sample_count == 0 ? 1 : profile->sample_count;
}

void model_io_prepare_input(TfLiteTensor* input, size_t sample_index) {
  const ModelProfile* profile = active_profile();
  if (input == 0) {
    puts("Input tensor is not available.");
    return;
  }
  if (profile->prepare_input != 0) {
    profile->prepare_input(input, sample_index);
    return;
  }

  if (profile->input_data != 0 && profile->input_bytes == input->bytes) {
    memcpy(input->data.int8, profile->input_data, input->bytes);
    printf("Input: using fixture data (%u bytes).\n", (unsigned)input->bytes);
  } else {
    memset(input->data.int8, 0, input->bytes);
    if (profile->input_data == 0) {
      printf("Input: no fixture for profile; using zeros (%u bytes).\n",
             (unsigned)input->bytes);
    } else {
      printf("Input: fixture is %u bytes, tensor is %u bytes; using zeros.\n",
             (unsigned)profile->input_bytes, (unsigned)input->bytes);
    }
  }
}

void model_io_verify_output(const TfLiteTensor* output, size_t sample_index) {
  const ModelProfile* profile = active_profile();
  if (output == 0) {
    puts("Output tensor is not available.");
    return;
  }
  if (profile->verify_output != 0) {
    profile->verify_output(output, sample_index);
    return;
  }

  if (profile->expected_output == 0) {
    puts("Output: no expected output for profile; verification skipped.");
    return;
  }

  if (profile->expected_output_bytes != output->bytes) {
    printf("Output: fixture is %u bytes, tensor is %u bytes; verification skipped.\n",
           (unsigned)profile->expected_output_bytes, (unsigned)output->bytes);
    return;
  }

  int error_count = 0;
  for (size_t i = 0; i < output->bytes; ++i) {
    int8_t actual = output->data.int8[i];
    int8_t expected = profile->expected_output[i];
    int diff = abs_int((int)actual - (int)expected);
    printf("Idx %u: HW=%d, Golden=%d\n", (unsigned)i, actual, expected);
    
    // 如果底層也有用到 profile->tolerance 比較，這裡轉成 int 比較
    if (diff > (int)profile->tolerance) {
      ++error_count;
    }
  }

  if (error_count == 0) {
    int tol_int = (int)profile->tolerance;
    int tol_frac = (int)((profile->tolerance - tol_int) * 1000000.0f);
    printf("[Status] ALL PASS (diff <= %d.%06d accepted).\n", tol_int, tol_frac);
  } else {
    printf("[Status] FAIL with %d errors.\n", error_count);
  }
}