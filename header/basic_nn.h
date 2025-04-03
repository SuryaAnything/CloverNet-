#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#define float32_t float
#define float64_t double

// Utility function
float64_t get_random_range(float64_t lower, float64_t upper);

// Struct for a pair of integers (layer dimensions)
typedef struct __int_pair {
    int64_t first;
    int64_t second;
} int64_pair_t;

#define fc_layer(a, b) (int64_pair_t){a, b}

// Neural network layer (fully connected layer)
typedef struct __neural_unit {
    float64_t* ze_vec;
    float64_t** wt_vec;
    int64_pair_t dim;
    float64_t (*acti_func)(float64_t);
    bool is_linear;
} neural_unit;

// Input/Output layer (no weights)
typedef struct __nerual_arr_layer {
    float64_t* ze_vec;
    int64_t size;
} neural_arr;

// Neural network module (full model)
typedef struct __neural_module {
    neural_arr input_layer;
    neural_unit* hidden_layer;
    int64_t num_hidden_layers;
    int64_t hidden_size;
    neural_arr output_layer;
} neural_module;

// Function declarations
void neural_unit_init(neural_unit* nunit, int64_t x_dim, int64_t y_dim, float64_t(*afunc)(float64_t), bool is_linear, float64_t range_s, float64_t range_e);
void neural_unit_erase_values(neural_unit* nunit);
void neural_module_init(neural_module* model, int64_t input_size, int64_t output_size, int64_t num_hidden_layers);
void neural_module_add_linear(neural_module* model, int64_pair_t dims);
void neural_module_add_nonlinear(neural_module* model, float64_t(*afunc)(float64_t));
void neural_module_validate(neural_module* model);
void neural_module_feed_forward(neural_module* model, float64_t* input);
void neural_module_back_prop(neural_module* model, float64_t* target, float64_t alpha);
void neural_module_reset_vals(neural_module* model);

// Activation functions
float64_t relu(float64_t x);

#endif // NEURAL_NET_H
