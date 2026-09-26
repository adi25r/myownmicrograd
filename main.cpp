#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cassert>
#include <memory>
#include "value.h"
#include "nn.h"

const double TOL = 1e-4;

// Test basic Value operations
void test_basic_ops() {
  std::cout << "\n=== Test Basic Operations ===" << std::endl;

  auto a = make_value(3.0);
  auto b = make_value(4.0);
  auto c = a + b;
  std::cout << "a + b = " << a->data << " + " << b->data << " = " << c->data << std::endl;

  auto d = a * b;
  std::cout << "a * b = " << a->data << " * " << b->data << " = " << d->data << std::endl;

  auto e = a - b;
  std::cout << "a - b = " << a->data << " - " << b->data << " = " << e->data << std::endl;

  auto f = pow(a, 2.0);
  std::cout << "a^2 = " << a->data << "^2 = " << f->data << std::endl;

  auto g = relu(make_value(-1.0));
  std::cout << "relu(-1) = " << g->data << std::endl;

  auto h = relu(make_value(2.0));
  std::cout << "relu(2) = " << h->data << std::endl;
}

// Test backward pass correctness
void test_backward() {
  std::cout << "\n=== Test Backward Pass ===" << std::endl;

  // Test: f(x) = x^2, df/dx = 2*x
  auto x = make_value(3.0);
  auto y = pow(x, 2.0);
  backward(y);

  double analytical = x->grad;
  double expected = 2.0 * 3.0;
  std::cout << "f(x) = x^2 at x=3: grad = " << analytical << " (expected " << expected << ")" << std::endl;
  assert(std::abs(analytical - expected) < TOL);

  // Test: f(x, y) = x*y + x, df/dx = y + 1
  auto x2 = make_value(2.0);
  auto y2 = make_value(3.0);
  auto z = x2 * y2 + x2;
  backward(z);

  std::cout << "f(x,y) = x*y + x at x=2, y=3: dx = " << x2->grad << " (expected 4.0)" << std::endl;
  assert(std::abs(x2->grad - 4.0) < TOL);
  assert(std::abs(y2->grad - 2.0) < TOL);
}

// Test ReLU activation
void test_relu() {
  std::cout << "\n=== Test ReLU ===" << std::endl;

  auto x1 = make_value(2.0);
  auto y1 = relu(x1);
  backward(y1);
  std::cout << "ReLU(2.0): output = " << y1->data << ", grad = " << x1->grad << std::endl;
  assert(std::abs(y1->data - 2.0) < TOL);
  assert(std::abs(x1->grad - 1.0) < TOL);

  auto x2 = make_value(-2.0);
  auto y2 = relu(x2);
  backward(y2);
  std::cout << "ReLU(-2.0): output = " << y2->data << ", grad = " << x2->grad << std::endl;
  assert(std::abs(y2->data - 0.0) < TOL);
  assert(std::abs(x2->grad - 0.0) < TOL);
}

// Test simple neuron
void test_neuron() {
  std::cout << "\n=== Test Single Neuron ===" << std::endl;

  // Create a simple neuron: 2 inputs, no nonlinearity for now
  Neuron neuron(2, false);

  // Manual weights for predictable output
  neuron.w[0] = make_value(0.5);
  neuron.w[1] = make_value(0.3);
  neuron.b = make_value(0.1);

  std::vector<std::shared_ptr<Value>> inputs = {make_value(2.0), make_value(3.0)};
  auto output = neuron(inputs);

  // Expected: 0.5*2 + 0.3*3 + 0.1 = 1.0 + 0.9 + 0.1 = 2.0
  std::cout << "Neuron output: " << output->data << " (expected 2.0)" << std::endl;
  assert(std::abs(output->data - 2.0) < TOL);

  // Test backward
  backward(output);
  std::cout << "Gradient w.r.t w[0]: " << neuron.w[0]->grad << " (expected 2.0)" << std::endl;
  std::cout << "Gradient w.r.t w[1]: " << neuron.w[1]->grad << " (expected 3.0)" << std::endl;
  assert(std::abs(neuron.w[0]->grad - 2.0) < TOL);
  assert(std::abs(neuron.w[1]->grad - 3.0) < TOL);
}

// Test layer
void test_layer() {
  std::cout << "\n=== Test Layer ===" << std::endl;

  Layer layer(2, 3, false);

  // Set fixed weights for testing
  for (size_t i = 0; i < layer.neurons.size(); ++i) {
    for (size_t j = 0; j < layer.neurons[i].w.size(); ++j) {
      layer.neurons[i].w[j] = make_value(0.1 * (i + 1) * (j + 1));
    }
    layer.neurons[i].b = make_value(0.0);
  }

  std::vector<std::shared_ptr<Value>> inputs = {make_value(1.0), make_value(2.0)};
  auto outputs = layer(inputs);

  std::cout << "Layer with 2 inputs, 3 outputs:" << std::endl;
  for (size_t i = 0; i < outputs.size(); ++i) {
    std::cout << "  Output " << i << ": " << outputs[i]->data << std::endl;
  }
  assert(outputs.size() == 3);
}

// Test MLP
void test_mlp() {
  std::cout << "\n=== Test MLP ===" << std::endl;

  std::vector<int> nouts = {3, 2};
  MLP mlp(2, nouts);

  std::cout << "MLP: " << mlp.repr() << std::endl;

  std::vector<std::shared_ptr<Value>> inputs = {make_value(1.0), make_value(2.0)};
  auto outputs = mlp(inputs);

  std::cout << "Input size: " << inputs.size() << std::endl;
  std::cout << "Output size: " << outputs.size() << std::endl;
  assert(outputs.size() == 2);
}

// Test simple regression
void test_regression() {
  std::cout << "\n=== Test Simple Regression ===" << std::endl;

  // Create a simple MLP: 1 input -> 2 hidden -> 1 output
  MLP net(1, {2, 1});

  // Generate training data: y = 2*x + 1
  std::vector<std::pair<double, double>> data = {
      {0.0, 1.0},
      {1.0, 3.0},
      {2.0, 5.0},
      {3.0, 7.0}
  };

  std::cout << "Training for 10 iterations..." << std::endl;

  for (int iter = 0; iter < 10; ++iter) {
    double total_loss = 0;

    for (const auto& [x, y_true] : data) {
      std::vector<std::shared_ptr<Value>> input = {make_value(x)};
      auto output = net(input);

      auto pred = output[0];
      auto loss = pow(pred - y_true, 2.0);

      backward(loss);
      total_loss += loss->data;
    }

    // Print loss every 2 iterations
    if (iter % 2 == 0) {
      std::cout << "Iteration " << iter << ": Loss = " << std::fixed << std::setprecision(6) << total_loss << std::endl;
    }

    // Gradient descent step
    auto params = net.parameters();
    for (auto& p : params) {
      p->data -= 0.01 * p->grad;
    }

    // Zero gradients for next iteration
    net.zero_grad();
  }

  // Test on new data point
  std::vector<std::shared_ptr<Value>> test_input = {make_value(1.5)};
  auto test_output = net(test_input);
  std::cout << "Prediction for x=1.5: " << test_output[0]->data << " (expected ~4.0)" << std::endl;
}

int main() {
  std::cout << "=== Micrograd C++ Test Suite ===" << std::endl;

  try {
    test_basic_ops();
    test_backward();
    test_relu();
    test_neuron();
    test_layer();
    test_mlp();
    test_regression();

    std::cout << "\n✓ All tests passed!" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "✗ Test failed: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
