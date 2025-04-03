#include "./header/basic_nn.h"
#include <time.h>
#include <stdio.h>

int main() {
    // Seed the random number generator
    srand(time(NULL));

    // Initialize the neural module:
    // - Input layer of size 3, output layer of size 2,
    // - Reserve space for up to 3 hidden layers.
    neural_module model;
    neural_module_init(&model, 3, 2, 3);

    // Build the network:
    neural_module_add_linear(&model, fc_layer(3, 64));
    neural_module_add_nonlinear(&model, relu);
    neural_module_add_linear(&model, fc_layer(64, 2));
    neural_module_add_nonlinear(&model, relu);

    // Validate the network's structure.
    neural_module_validate(&model);

    float64_t input[3] = {0.5, -0.3, 1.2};
    float64_t target[2] = {0.8, 0.4};

    // Choose a learning rate.
    float64_t alpha = 0.01;

    for (int epoch = 0; epoch < 50; epoch++) {
        neural_module_feed_forward(&model, input);

        float64_t diff_sum = 0.0;
        for (int i = 0; i < model.output_layer.size; i++) {
            diff_sum += (model.output_layer.ze_vec[i] - target[i]);
        }

        // Print epoch details.
        printf("Epoch %d:\n", epoch);
        printf("  Output: ");
        for (int i = 0; i < model.output_layer.size; i++) {
            printf("%lf ", model.output_layer.ze_vec[i]);
        }
        printf("\n  Loss differential: %lf\n", diff_sum);

        // Perform backpropagation using the target values and learning rate.
        neural_module_back_prop(&model, target, alpha);
    }

    // Final feed-forward to see the final updated output.
    neural_module_feed_forward(&model, input);
    printf("Final Output: ");
    for (int i = 0; i < model.output_layer.size; i++) {
        printf("%lf ", model.output_layer.ze_vec[i]);
    }
    
    printf("\nExpected Output: ");
    for (int i = 0; i < model.output_layer.size; i++) {
        printf("%lf ", target[i]);
    }
    printf("\n");

    return 0;
}
