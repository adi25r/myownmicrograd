#include <cmath>
#include <unordered_set>
#include "value.h"

Value::Value(double data, std::vector<std::shared_ptr<Value>> children, std::string op)
    : data{data}, grad{0}, children{std::move(children)}, op{std::move(op)}, _backward([](){}) {}

std::shared_ptr<Value> make_value(double data, std::vector<std::shared_ptr<Value>> children, std::string op) {
  return std::make_shared<Value>(data, std::move(children), std::move(op));
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
  auto out = make_value(a->data + b->data, {a, b}, "+");

  out->_backward = [a, b, out]() {
    a->grad += out->grad;
    b->grad += out->grad;
  };

  return out;
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, double b) {
  auto out = make_value(a->data + b, {a}, "+");

  out->_backward = [a, out]() {
    a->grad += out->grad;
  };

  return out;
}

std::shared_ptr<Value> operator+(double a, const std::shared_ptr<Value>& b) {
  return b + a;
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
  auto out = make_value(a->data * b->data, {a, b}, "*");

  out->_backward = [a, b, out]() {
    a->grad += b->data * out->grad;
    b->grad += a->data * out->grad;
  };

  return out;
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, double b) {
  auto out = make_value(a->data * b, {a}, "*");

  out->_backward = [a, b, out]() {
    a->grad += b * out->grad;
  };

  return out;
}

std::shared_ptr<Value> operator*(double a, const std::shared_ptr<Value>& b) {
  return b * a;
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a) {
  return a * -1.0;
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
  return a + (-b);
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, double b) {
  return a + (-b);
}

std::shared_ptr<Value> operator-(double a, const std::shared_ptr<Value>& b) {
  return (-b) + a;
}

std::shared_ptr<Value> pow(const std::shared_ptr<Value>& a, double other) {
  auto out = make_value(std::pow(a->data, other), {a}, "**");

  out->_backward = [a, other, out]() {
    a->grad += (other * std::pow(a->data, other - 1)) * out->grad;
  };

  return out;
}

std::shared_ptr<Value> relu(const std::shared_ptr<Value>& a) {
  auto out = make_value(a->data > 0 ? a->data : 0, {a}, "ReLU");

  out->_backward = [a, out]() {
    a->grad += (out->data > 0 ? 1.0 : 0.0) * out->grad;
  };

  return out;
}

void backward(const std::shared_ptr<Value>& root) {
  std::vector<std::shared_ptr<Value>> topo;
  std::unordered_set<Value*> visited;

  std::function<void(const std::shared_ptr<Value>&)> build_topo = [&](const std::shared_ptr<Value>& v) {
    if (visited.find(v.get()) == visited.end()) {
      visited.insert(v.get());
      for (const auto& child : v->children) {
        build_topo(child);
      }
      topo.push_back(v);
    }
  };

  build_topo(root);

  root->grad = 1.0;
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    (*it)->_backward();
  }
}
