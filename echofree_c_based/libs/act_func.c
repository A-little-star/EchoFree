#include "../include/act_func.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Sigmoid 激活函数
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Tanh 激活函数
float tanh_activation(float x) {
    return tanhf(x);
}