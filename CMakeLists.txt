cmake_minimum_required(VERSION 4.2)
project(NeuralNetworkHeatMap)

set(CMAKE_CXX_STANDARD 23)

add_executable(NeuralNetworkHeatMap main.cpp
        neural_network.h)
