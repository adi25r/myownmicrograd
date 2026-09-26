#pragma once

#include <string>
#include <vector>
#include <functional>
#include <memory>

class Value {
public:
  double data;
  double grad;

  std::vector<std::shared_ptr<Value>> children;
  std::string op;
  std::function<void()> _backward;

  Value(double data, std::vector<std::shared_ptr<Value>> children = {}, std::string op = "");
};

std::shared_ptr<Value> make_value(double data, std::vector<std::shared_ptr<Value>> children = {}, std::string op = "");

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, double b);
std::shared_ptr<Value> operator+(double a, const std::shared_ptr<Value>& b);

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, double b);
std::shared_ptr<Value> operator*(double a, const std::shared_ptr<Value>& b);

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, double b);
std::shared_ptr<Value> operator-(double a, const std::shared_ptr<Value>& b);

std::shared_ptr<Value> pow(const std::shared_ptr<Value>& a, double other);
std::shared_ptr<Value> relu(const std::shared_ptr<Value>& a);

void backward(const std::shared_ptr<Value>& root);
