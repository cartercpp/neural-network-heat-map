#include <iostream>
#include <format>
#include <chrono>
#include <random>
#include <thread>
#include <stop_token>
#include <cmath>
#include "neural_network.h"

int main()
{
    constexpr int rows = 60,
                  columns = 120,
                  minRed = 125,
                  maxRed = 255;

    constexpr double minX = -16,
                     maxX = 16,
                     minY = -17,
                     maxY = 12,
                     maxT = 3.14159 * 2;

    char grid[rows][columns];

    auto resetGrid = [&grid]()
    {
        for (int row = 0; row < rows; ++row)
            for (int column = 0; column < columns; ++column)
                grid[row][column] = ' ';
    };

    auto moveCursor = [](int row, int column)
    {
        std::cout << std::format("\033[{};{}H", row + 1, column + 1);
    };

    auto changeColor = [](int red, int green, int blue)
    {
        std::cout << std::format("\033[38;2;{};{};{}m", red, green, blue);
    };

    std::cout << "\033[2J";
    std::cout << "\033[?25l";

    std::random_device rd;
    std::uniform_real_distribution<double> tDist(0, maxT);
    neural_network<1, 20, 20, 20, 20, 2> heartNn(0.01);

    const auto& weights{heartNn.weights()};

    std::jthread thr{[&](std::stop_token st)
    {
        while (!st.stop_requested())
        {
            resetGrid();

            for (double tValue = 0; tValue < maxT; tValue += 0.01)
            {
                const double tNormalized = tValue / maxT;

                const auto prediction = heartNn.predict({tNormalized});
                const double predictedX = minX + prediction[0] * (maxX - minX),
                             predictedY = minY + prediction[1] * (maxY - minY);

                const int row = rows - 1 - static_cast<int>((predictedY - minY) / (maxY - minY) * (rows - 1)),
                          column = static_cast<int>((predictedX - minX) / (maxX - minX) * (columns - 1));

                if ((row >= 0) && (row < rows) && (column >= 0) && (column < columns))
                    grid[row][column] = 'o';
            }

            changeColor(255, 0, 0);
            for (int row = 0; row < rows; ++row)
            {
                moveCursor(row, 0);

                for (int column = 0; column < columns; ++column)
                    std::cout << grid[row][column];
            }

            const auto oldWeights = weights;

            for (int iter = 0; iter < 100; ++iter)
            {
                const double tValue = tDist(rd),
                             tNormalized = tValue / maxT;

                const double xValue = 16 * std::pow(std::sin(tValue), 3),
                             yValue = 13 * std::cos(tValue) - 5 * std::cos(2 * tValue) - 2 * std::cos(3 * tValue)
                                      - std::cos(4 * tValue);

                const double xNormalized = (xValue - minX) / (maxX - minX),
                             yNormalized = (yValue - minY) / (maxY - minY);

                heartNn.fit({tNormalized}, {xNormalized, yNormalized});
            }

            {
                double minChange = std::numeric_limits<double>::max(),
                       maxChange = std::numeric_limits<double>::min();

                for (int i = 0; i < weights.size(); ++i)
                    for (int weightRow = 0; weightRow < weights[i].size(); ++weightRow)
                        for (int weightColumn = 0; weightColumn < weights[i][weightRow].size(); ++weightColumn)
                        {
                            const double weight = weights[i][weightRow][weightColumn],
                                         oldWeight = oldWeights[i][weightRow][weightColumn],
                                         weightChange = weight - oldWeight;

                            minChange = std::min(minChange, weightChange);
                            maxChange = std::max(maxChange, weightChange);
                        }

                int row = 0;
                for (int i = 0; i < weights.size(); ++i)
                    for (int weightRow = 0; weightRow < weights[i].size(); ++weightRow)
                    {
                        moveCursor(row, columns + 10);

                        for (int weightColumn = 0; weightColumn < weights[i][weightRow].size(); ++weightColumn)
                        {
                            const double weight = weights[i][weightRow][weightColumn],
                                         oldWeight = oldWeights[i][weightRow][weightColumn],
                                         weightChange = weight - oldWeight,
                                         weightChangeNormalized = (weightChange - minChange) / (maxChange - minChange);

                            const int red = static_cast<int>(minRed + weightChangeNormalized * (minRed - maxRed));
                            changeColor(red, 0, 0);
                            std::cout << "o ";
                        }

                        ++row;
                    }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }};
    std::cin.get();

    std::cout << "\033[?25h";
    std::cout << "\033[0m";
}