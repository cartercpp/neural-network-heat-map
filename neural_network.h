//
// Created by cartercpp on 7/16/26.
//

#ifndef NEURALNETWORKHEATMAP_NEURAL_NETWORK_H
#define NEURALNETWORKHEATMAP_NEURAL_NETWORK_H

#include <stdexcept>
#include <utility>
#include <random>
#include <vector>
#include <array>
#include <cmath>

template <int... Is>
class neural_network
{
private:

    using WeightMatrix = std::vector<std::vector<double>>;

public:

    // CONSTRUCTORS

    explicit neural_network(double learningRate)
        : m_learningRate{learningRate}
    {
        m_weightMatrices.reserve(s_layers);

        std::random_device rd;
        std::uniform_real_distribution<double> weightDist(-0.5, 0.5);

        for (int layer = 1; layer < s_layers; ++layer)
        {
            const int layerSize = s_neuronsPerLayer[layer],
                      prevLayerSize = s_neuronsPerLayer[layer - 1];

            WeightMatrix weights(layerSize, std::vector<double>(prevLayerSize, 0));
            for (int i = 0; i < layerSize; ++i)
                for (int i2 = 0; i2 < prevLayerSize; ++i2)
                    weights[i][i2] = weightDist(rd);

            m_weightMatrices.emplace_back(std::move(weights));
        }
    }

    // METHODS

    const std::vector<WeightMatrix>& weights() const noexcept
    {
        return m_weightMatrices;
    }

    std::vector<double> predict(const std::vector<double>& input) const
    {
        return Forward(input).back();
    }

    void fit(const std::vector<double>& input, const std::vector<double>& target)
    {
        Fit(input, target);
    }

private:

    static constexpr int s_layers = static_cast<int>(sizeof...(Is));
    static_assert(s_layers >= 3, "Must have at least an input layer, a hidden layer, and an output layer");
    static_assert(((Is > 0) && ...), "Each layer must have at least one activation");
    static constexpr std::array<int, s_layers> s_neuronsPerLayer{Is...};

    static double Sech(double x)
    {
        return 1 / std::cosh(x);
    }

    static double TanhDerivative(double x)
    {
        return Sech(x) * Sech(x);
    }

    std::vector<std::vector<double>> Forward(const std::vector<double>& input) const
    {
        if (input.size() != s_neuronsPerLayer[0])
            throw std::invalid_argument{"Incorrect # of inputs given"};

        std::vector<std::vector<double>> activations;
        activations.reserve(s_layers);
        activations.push_back(input);

        for (int layer = 0; layer < s_layers - 1; ++layer)
        {
            const std::vector<double>& sourceActivations{activations[layer]};
            const WeightMatrix& sourceWeights{m_weightMatrices[layer]};
            const int sourceSize = s_neuronsPerLayer[layer];

            const int destSize = s_neuronsPerLayer[layer + 1];
            std::vector<double> destActivations(destSize, 0);

            for (int destNeuron = 0; destNeuron < destSize; ++destNeuron)
            {
                double weightedSum = 0;

                for (int sourceNeuron = 0; sourceNeuron < sourceSize; ++sourceNeuron)
                    weightedSum += sourceActivations[sourceNeuron] * sourceWeights[destNeuron][sourceNeuron];

                destActivations[destNeuron] = std::tanh(weightedSum);
            }

            activations.emplace_back(std::move(destActivations));
        }

        return activations;
    }

    void Fit(const std::vector<double>& input, const std::vector<double>& target)
    {
        const int numberOfOutputs = s_neuronsPerLayer[s_layers - 1];
        if (target.size() != numberOfOutputs)
            throw std::invalid_argument{"Incorrect # of targets given"};

        const std::vector<std::vector<double>> activations{Forward(input)};
        const std::vector<double>& prediction{activations.back()};

        std::vector<double> delta(numberOfOutputs, 0);
        for (int i = 0; i < numberOfOutputs; ++i)
            delta[i] = (prediction[i] - target[i]) * TanhDerivative(prediction[i]);

        for (int layer = s_layers - 2; layer >= 0; --layer)
        {
            const std::vector<double>& sourceActivations{activations[layer]};
            WeightMatrix& sourceWeightsRef{m_weightMatrices[layer]};
            const int sourceSize = s_neuronsPerLayer[layer];
            const int destSize = s_neuronsPerLayer[layer + 1];

            std::vector<double> newDelta(sourceSize, 0);
            for (int sourceNeuron = 0; sourceNeuron < sourceSize; ++sourceNeuron)
            {
                double cost = 0;

                for (int destNeuron = 0; destNeuron < destSize; ++destNeuron)
                    cost += delta[destNeuron] * sourceWeightsRef[destNeuron][sourceNeuron];

                if (layer > 0)
                    cost *= TanhDerivative(sourceActivations[sourceNeuron]);

                newDelta[sourceNeuron] = cost;
            }

            for (int destNeuron = 0; destNeuron < destSize; ++destNeuron)
                for (int sourceNeuron = 0; sourceNeuron < sourceSize; ++sourceNeuron)
                    sourceWeightsRef[destNeuron][sourceNeuron]
                        -= m_learningRate * delta[destNeuron] * sourceActivations[sourceNeuron];

            std::swap(delta, newDelta);
        }
    }

    std::vector<WeightMatrix> m_weightMatrices;
    double m_learningRate;
};

#endif //NEURALNETWORKHEATMAP_NEURAL_NETWORK_H