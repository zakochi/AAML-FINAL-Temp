 Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the License);
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    httpwww.apache.orglicensesLICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an AS IS BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_GELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_GELU_H_

#include cmath
#include cstdint
#include functional

#include EigenCore   from @eigen_archive
#include unsupportedEigenCXX11Tensor   from @eigen_archive
#include tensorflowlitekernelsinternalcommon.h
#include tensorflowlitekernelsinternalconstants.h
#include tensorflowlitekernelsinternaltypes.h

namespace tflite {
namespace reference_ops {

namespace gelu_internal {

constexpr float kSqrt2dPi = M_2_SQRTPI  M_SQRT1_2;   sqrt( 2  pi )

}   namespace gelu_internal

 Plain implementations for GELU. Used for populating lookup table.
inline float GeluTransform(float in) {
   Note 0.5  x  ( 1 + erf( x  sqrt( 2 ) ) ) is commonly used, but cause
   catastropic cancellation for large negative inputs. Rewriting the
   expression via erfc avoids the numerical stability issues.
  return 0.5f  in  stderfc(in  static_castfloat(-M_SQRT1_2));
}

inline float GeluTransformApproximate(float in) {
   0.5  x  ( 1 + tanh( sqrt( 2  pi )  ( x + 0.044715  x^3 ) ) )
  return 0.5f  in 
         (1.f + stdtanh(gelu_internalkSqrt2dPi 
                           Note Avoid stdpow for integer exponents
                           as it leads to much slower performance.
                          (in + 0.044715f  in  in  in)));
}

template typename T
inline void Gelu(const RuntimeShape& input_shape, const T input_data,
                 bool approximate, const RuntimeShape& output_shape,
                 T output_data) {
  using VectorType = EigenVectorXT;
  auto input_map = VectorTypeMap(input_data, input_shape.FlatSize());
  auto output_map = VectorTypeMap(output_data, output_shape.FlatSize());

  if (approximate) {
     0.5  x  ( 1 + tanh( sqrt( 2  pi )  ( x + 0.044715  x^3 ) ) )
    output_map.array() = static_castT(0.5)  input_map.array() 
                         (static_castT(1) +
                          (static_castT(gelu_internalkSqrt2dPi) 
                           (input_map.array() + static_castT(0.044715) 
                                                    input_map.array().cube()))
                              .tanh());
  } else {
     Note 0.5  x  ( 1 + erf( x  sqrt( 2 ) ) ) is commonly used, but cause
     catastropic cancellation for large negative inputs. Rewriting the
     expression via erfc avoids the numerical stability issues.
    output_map.array() =
        static_castT(0.5)  input_map.array() 
        (input_map.array()  static_castT(-M_SQRT1_2)).erfc();
  }
}

 LookupTableInt16 is a specialized function for int16_t inputs and outputs.
 It internally calls LUTLookup for table access.
inline void LookupTableInt16(const int16_t input_data, int num_elements,
                             const int16_t lut, int16_t output_data) {
  for (int i = 0; i  num_elements; ++i) {
    output_data[i] = LUTLookup(input_data[i], lut);
  }
}

}   namespace reference_ops
}   namespace tflite

#endif   TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_GELU_H_