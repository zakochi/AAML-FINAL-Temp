#include "tflm_ops.h"

void tflm_register_project_ops(ProjectOpResolver* resolver) {
    resolver->AddAdd();
    resolver->AddBatchMatMul();
    resolver->AddCast();
    resolver->AddDequantize();
    resolver->AddFullyConnected();
    resolver->AddGather();
    resolver->AddMean();
    resolver->AddMul();
    resolver->AddQuantize();
    resolver->AddReshape();
    resolver->AddRsqrt();
    resolver->AddSoftmax();
    resolver->AddSquaredDifference();
    resolver->AddStridedSlice();
    resolver->AddSub();
    resolver->AddTanh();
    resolver->AddTranspose();
    resolver->AddRelu();
}