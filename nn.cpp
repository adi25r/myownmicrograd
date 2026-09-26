#include "nn.h"
#include <iostream>

std::mt19937 Neuron::rng(std::random_device{}());
std::uniform_real_distribution<double> Neuron::dist(-1.0, 1.0);

// Module base class
void Module::zero_grad() {
  auto params = parameters();
  for (auto& p : params) {
    p->grad = 0;
  }
}

std::vector<std::shared_ptr<Value>> Module::parameters() {
  return {};
}

// Neuron class
Neuron::Neuron(int nin, bool nonlin) : b(make_value(0)), nonlin(nonlin) {
  w.reserve(nin);
  for (int i = 0; i < nin; ++i) {
    w.push_back(make_value(dist(rng)));
  }
}

std::shared_ptr<Value> Neuron::operator()(const std::vector<std::shared_ptr<Value>>& x) {
  std::shared_ptr<Value> act = b;
  for (size_t i = 0; i < w.size(); ++i) {
    act = act + (w[i] * x[i]);
  }
  return nonlin ? relu(act) : act;
}

std::vector<std::shared_ptr<Value>> Neuron::parameters() {
  std::vector<std::shared_ptr<Value>> params = w;
  params.push_back(b);
  return params;
}

std::string Neuron::repr() {
  std::ostringstream oss;
  oss << (nonlin ? "ReLU" : "Linear") << "Neuron(" << w.size() << ")";
  return oss.str();
}

// Layer class
Layer::Layer(int nin, int nout, bool nonlin) {
  neurons.reserve(nout);
  for (int i = 0; i < nout; ++i) {
    neurons.emplace_back(nin, nonlin);
  }
}

std::vector<std::shared_ptr<Value>> Layer::operator()(const std::vector<std::shared_ptr<Value>>& x) {
  std::vector<std::shared_ptr<Value>> out;
  out.reserve(neurons.size());
  for (auto& neuron : neurons) {
    out.push_back(neuron(x));
  }
  return out;
}

std::vector<std::shared_ptr<Value>> Layer::parameters() {
  std::vector<std::shared_ptr<Value>> params;
  for (auto& neuron : neurons) {
    auto neuron_params = neuron.parameters();
    params.insert(params.end(), neuron_params.begin(), neuron_params.end());
  }
  return params;
}

std::string Layer::repr() {
  std::ostringstream oss;
  oss << "Layer of [";
  for (size_t i = 0; i < neurons.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << neurons[i].repr();
  }
  oss << "]";
  return oss.str();
}

// MLP class
MLP::MLP(int nin, const std::vector<int>& nouts) {
  std::vector<int> sizes;
  sizes.push_back(nin);
  sizes.insert(sizes.end(), nouts.begin(), nouts.end());

  layers.reserve(nouts.size());
  for (size_t i = 0; i < nouts.size(); ++i) {
    bool nonlin = (i != nouts.size() - 1);
    layers.emplace_back(sizes[i], sizes[i + 1], nonlin);
  }
}

std::vector<std::shared_ptr<Value>> MLP::operator()(const std::vector<std::shared_ptr<Value>>& x) {
  std::vector<std::shared_ptr<Value>> out = x;
  for (auto& layer : layers) {
    out = layer(out);
  }
  return out;
}

std::vector<std::shared_ptr<Value>> MLP::parameters() {
  std::vector<std::shared_ptr<Value>> params;
  for (auto& layer : layers) {
    auto layer_params = layer.parameters();
    params.insert(params.end(), layer_params.begin(), layer_params.end());
  }
  return params;
}

std::string MLP::repr() {
  std::ostringstream oss;
  oss << "MLP of [";
  for (size_t i = 0; i < layers.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << layers[i].repr();
  }
  oss << "]";
  return oss.str();
}
