#include "../header/basic_nn.h"

float64_t get_random_range(float64_t lower, float64_t upper) {
    return ((float64_t)rand() / RAND_MAX) * (upper - lower) + lower;
}

void neural_unit_init(neural_unit *nunit, int64_t x_dim, int64_t y_dim, float64_t (*afunc)(float64_t), bool is_linear, float64_t range_s, float64_t range_e) {
    // x_dim corresponds to the number of inputs (from previous layer)
    // y_dim corresponds to the number of neurons in this layer
    nunit->is_linear = is_linear;
    nunit->acti_func = afunc;
    nunit->ze_vec = calloc(y_dim, sizeof(float64_t));
    nunit->wt_vec = malloc(x_dim * sizeof(float64_t*));
    for (int64_t i = 0; i < x_dim; ++i) {
        nunit->wt_vec[i] = malloc(y_dim * sizeof(float64_t));
        for (int64_t j = 0; j < y_dim; ++j) {
            nunit->wt_vec[i][j] = get_random_range(range_s, range_e);
        }
    }
    nunit->dim.first = x_dim;
    nunit->dim.second = y_dim;
}

void neural_unit_erase_values(neural_unit* nunit) {
    for (int64_t i = 0; i < nunit->dim.second; ++i) {
        nunit->ze_vec[i] = 0.0;
    }
}

void neural_module_init(neural_module* model, int64_t input_size, int64_t output_size, int64_t num_hidden_layers) {
    model->hidden_layer = malloc(num_hidden_layers * sizeof(neural_unit));
    model->num_hidden_layers = num_hidden_layers;
    model->input_layer.size = input_size;
    model->input_layer.ze_vec = calloc(input_size, sizeof(float64_t));
    model->output_layer.size = output_size;
    model->hidden_size = 0;
    model->output_layer.ze_vec = calloc(output_size, sizeof(float64_t));
}

void neural_module_add_linear(neural_module* model, int64_pair_t dims) {
    if (model->hidden_size < model->num_hidden_layers) {
        neural_unit_init(&(model->hidden_layer[model->hidden_size]), dims.first, dims.second, NULL, true, 0, 1);
        model->hidden_size++;
    }
}

void neural_module_add_nonlinear(neural_module* model, float64_t(*afunc)(float64_t)) {
    if (model->hidden_size < model->num_hidden_layers) {
        // For a non-linear layer, the x_dim is not used (set to 1) since weights are not needed.
        if (model->hidden_size == 0)
            neural_unit_init(&(model->hidden_layer[model->hidden_size]), 1, model->input_layer.size, afunc, false, 0.1, 1);
        else
            neural_unit_init(&(model->hidden_layer[model->hidden_size]), 1, model->hidden_layer[model->hidden_size - 1].dim.second, afunc, false, 0.1, 1);
        model->hidden_size++;
    }
}

void neural_module_validate(neural_module* model) {
    if (model->input_layer.ze_vec == NULL) {
        fprintf(stderr, "Error: Input layer not initialized.\n");
        exit(EXIT_FAILURE);
    }

    for (int64_t i = 0; i < model->hidden_size; i++) {
        neural_unit* layer = &model->hidden_layer[i];

        if (layer->ze_vec == NULL) {
            fprintf(stderr, "Error: Hidden layer %lld has uninitialized neuron vector.\n", (long long)i);
            exit(EXIT_FAILURE);
        }

        if (layer->is_linear) {
            if (layer->wt_vec == NULL) {
                fprintf(stderr, "Error: Linear layer %lld has uninitialized weight matrix.\n", (long long)i);
                exit(EXIT_FAILURE);
            }
            // Check each weight row
            for (int64_t j = 0; j < layer->dim.first; j++) {
                if (layer->wt_vec[j] == NULL) {
                    fprintf(stderr, "Error: Weight row %lld in layer %lld is uninitialized.\n", (long long)j, (long long)i);
                    exit(EXIT_FAILURE);
                }
            }
            // For the first layer, check against input layer size
            if (i == 0) {
                if (layer->dim.first != model->input_layer.size) {
                    fprintf(stderr, "Error: Linear layer %lld expects input dim %lld but input layer has size %lld.\n",
                            (long long)i, (long long)layer->dim.first, (long long)model->input_layer.size);
                    exit(EXIT_FAILURE);
                }
            } else {
                if (model->hidden_layer[i - 1].dim.second != layer->dim.first) {
                    fprintf(stderr, "Error: Dimensional mismatch between layer %lld (output dim %lld) and layer %lld (input dim %lld).\n",
                            (long long)(i - 1), (long long)model->hidden_layer[i - 1].dim.second, (long long)i, (long long)layer->dim.first);
                    exit(EXIT_FAILURE);
                }
            }
        } else { // Nonlinear layer
            if (i == 0) {
                if (layer->dim.second != model->input_layer.size) {
                    fprintf(stderr, "Error: Nonlinear layer %lld expects dim %lld but input layer has size %lld.\n",
                            (long long)i, (long long)layer->dim.second, (long long)model->input_layer.size);
                    exit(EXIT_FAILURE);
                }
            } else {
                if (layer->dim.second != model->hidden_layer[i - 1].dim.second) {
                    fprintf(stderr, "Error: Nonlinear layer %lld has mismatched size (%lld vs %lld) compared to previous layer.\n",
                            (long long)i, (long long)layer->dim.second, (long long)model->hidden_layer[i - 1].dim.second);
                    exit(EXIT_FAILURE);
                }
            }
        }
    }

    // Validate output layer matches last hidden layer's output dimension (if hidden layers exist)
    if (model->hidden_size > 0) {
        neural_unit* last_layer = &model->hidden_layer[model->hidden_size - 1];
        if (last_layer->dim.second != model->output_layer.size) {
            fprintf(stderr, "Error: Output layer size (%lld) doesn't match last hidden layer output (%lld).\n",
                    (long long)model->output_layer.size, (long long)last_layer->dim.second);
            exit(EXIT_FAILURE);
        }
    }

    if (model->output_layer.ze_vec == NULL) {
        fprintf(stderr, "Error: Output layer not initialized.\n");
        exit(EXIT_FAILURE);
    }

    printf("Neural module validation passed.\n");
}

void neural_module_feed_forward(neural_module* model, float64_t* input) {
    // Copy input into the input layer
    for (int64_t i = 0; i < model->input_layer.size; i++) {
        model->input_layer.ze_vec[i] = input[i];
    }

    // Feed-forward through each hidden layer
    for (int64_t i = 0; i < model->hidden_size; i++) {
        neural_unit* curr_layer = &model->hidden_layer[i];
        float64_t* prev_output;
        int64_t prev_size;

        if (i == 0) {
            prev_output = model->input_layer.ze_vec;
            prev_size = model->input_layer.size;
        } else {
            neural_unit* prev_layer = &model->hidden_layer[i - 1];
            prev_output = prev_layer->ze_vec;
            prev_size = prev_layer->dim.second;
        }

        if (curr_layer->is_linear) {
            // Multiply previous layer output by weights
            for (int64_t j = 0; j < curr_layer->dim.second; j++) {
                float64_t sum = 0.0;
                for (int64_t k = 0; k < prev_size; k++) {
                    sum += prev_output[k] * curr_layer->wt_vec[k][j];
                }
                curr_layer->ze_vec[j] = sum;
            }
        } else {
            // Apply activation function element-wise
            for (int64_t j = 0; j < curr_layer->dim.second; j++) {
                curr_layer->ze_vec[j] = curr_layer->acti_func(prev_output[j]);
            }
        }
    }

    // Copy final hidden layer output to output layer
    if (model->hidden_size > 0) {
        neural_unit* last_layer = &model->hidden_layer[model->hidden_size - 1];
        if (last_layer->dim.second != model->output_layer.size) {
            fprintf(stderr, "Error: Output layer dimension mismatch.\n");
            exit(EXIT_FAILURE);
        }
        for (int64_t j = 0; j < model->output_layer.size; j++) {
            model->output_layer.ze_vec[j] = last_layer->ze_vec[j];
        }
    } else {
        if (model->input_layer.size != model->output_layer.size) {
            fprintf(stderr, "Error: Output layer dimension mismatch with input layer.\n");
            exit(EXIT_FAILURE);
        }
        for (int64_t j = 0; j < model->output_layer.size; j++) {
            model->output_layer.ze_vec[j] = model->input_layer.ze_vec[j];
        }
    }
}

void neural_module_reset_vals(neural_module* model) {
    for (int64_t i = 0; i < model->num_hidden_layers; ++i) {
        neural_unit_erase_values(&(model->hidden_layer[i]));
    }
}

void neural_module_back_prop(neural_module* model, float64_t* target, float64_t alpha) {
    float64_t h = 1e-5;
    // Allocate delta arrays for each hidden layer.
    float64_t** deltas = malloc(model->hidden_size * sizeof(float64_t*));
    for (int64_t i = 0; i < model->hidden_size; i++) {
        deltas[i] = calloc(model->hidden_layer[i].dim.second, sizeof(float64_t));
    }

    // Compute delta for the last hidden layer (assumed to be non-linear, matching output_layer).
    neural_unit* last = &model->hidden_layer[model->hidden_size - 1];
    for (int64_t i = 0; i < last->dim.second; i++) {
        float64_t output = model->output_layer.ze_vec[i];
        float64_t error = output - target[i];  // dL/dy for MSE loss
        float64_t d_act = 1.0;  // Default derivative for linear layers
        if (!last->is_linear && last->acti_func != NULL) {
            d_act = (last->acti_func(last->ze_vec[i] + h) - last->acti_func(last->ze_vec[i])) / h;
        }
        deltas[model->hidden_size - 1][i] = error * d_act;
    }

    // Backpropagate the error through hidden layers.
    for (int64_t layer = model->hidden_size - 2; layer >= 0; layer--) {
        neural_unit* curr = &model->hidden_layer[layer];
        neural_unit* next = &model->hidden_layer[layer + 1];
        for (int64_t i = 0; i < curr->dim.second; i++) {
            float64_t sum = 0.0;
            // If the next layer is linear, propagate error via its weight matrix.
            if (next->is_linear) {
                // For a linear layer, curr->dim.second should match next->dim.first.
                for (int64_t j = 0; j < next->dim.second; j++) {
                    sum += next->wt_vec[i][j] * deltas[layer + 1][j];
                }
            } else {
                // For a non-linear next layer, skip weight multiplication.
                sum = deltas[layer + 1][i];
            }
            float64_t d_act = 1.0;
            if (!curr->is_linear && curr->acti_func != NULL) {
                d_act = (curr->acti_func(curr->ze_vec[i] + h) - curr->acti_func(curr->ze_vec[i])) / h;
            }
            deltas[layer][i] = sum * d_act;
        }
    }

    for (int64_t layer = 0; layer < model->hidden_size; layer++) {
        neural_unit* curr = &model->hidden_layer[layer];
        if (!curr->is_linear) continue;
        int64_t prev_size = (layer == 0) ? model->input_layer.size : model->hidden_layer[layer - 1].dim.second;
        float64_t* prev_output = (layer == 0) ? model->input_layer.ze_vec : model->hidden_layer[layer - 1].ze_vec;
        for (int64_t i = 0; i < curr->dim.first; i++) {
            for (int64_t j = 0; j < curr->dim.second; j++) {
                curr->wt_vec[i][j] -= alpha * deltas[layer][j] * prev_output[i];
            }
        }
    }

    for (int64_t i = 0; i < model->hidden_size; i++) {
        free(deltas[i]);
    }
    free(deltas);
}

float64_t relu(float64_t x) {
    return (x > 0) ? x : 0;
}
