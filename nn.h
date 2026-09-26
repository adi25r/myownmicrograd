#pragma once

#include <vector>
#include <string>
#include <random>
#include <sstream>
#include <memory>
#include "value.h"

class Module {
public:
  virtual ~Module() = default;

  virtual void zero_grad();
  virtual std::vector<std::shared_ptr<Value>> parameters();
};

class Neuron : public Module {
public:
  std::vector<std::shared_ptr<Value>> w;
  std::shared_ptr<Value> b;
  bool nonlin;

  static std::mt19937 rng;
  static std::uniform_real_distribution<double> dist;
  Neuron(int nin, bool nonlin = true);

  std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& x);

  std::vector<std::shared_ptr<Value>> parameters() override;
  std::string repr();
};

class Layer : public Module {
public:
  std::vector<Neuron> neurons;
  Layer(int nin, int nout, bool nonlin = true);

  std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& x);

  std::vector<std::shared_ptr<Value>> parameters() override;
  std::string repr();
};

class MLP : public Module {
private:
  std::vector<Layer> layers;

public:
  MLP(int nin, const std::vector<int>& nouts);

  std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& x);

  std::vector<std::shared_ptr<Value>> parameters() override;
  std::string repr();
};
