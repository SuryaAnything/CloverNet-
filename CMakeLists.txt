cmake_minimum_required(VERSION 3.10)

# Project Name and Version
project(CLove_NN_C VERSION 0.1.0 LANGUAGES C)

# Create a library from the neural network implementation
add_library(clove_nn STATIC source/basic_nn.c)

# Create the executable from main.c
add_executable(CLove_NN_C main.c)

# Link the executable with the library
target_link_libraries(CLove_NN_C PRIVATE clove_nn)
