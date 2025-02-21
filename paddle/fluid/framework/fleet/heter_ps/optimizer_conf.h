/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

namespace paddle {
namespace framework {

class OptimizerConfig {
 public:
  float nonclk_coeff = 0.1;
  float clk_coeff = 1;

  float min_bound = -10;
  float max_bound = 10;
  float learning_rate = 0.05;
  float initial_g2sum = 3.0;
  float initial_range = 0;

  float mf_create_thresholds = 10;
  float mf_learning_rate = 0.05;
  float mf_initial_g2sum = 3.0;
  float mf_initial_range = 1e-4;
  float mf_min_bound = -10;
  float mf_max_bound = 10;

  void set_sparse_sgd(float nonclk_coeff,
                      float clk_coeff,
                      float min_bound,
                      float max_bound,
                      float learning_rate,
                      float initial_g2sum,
                      float initial_range) {
    this->nonclk_coeff = nonclk_coeff;
    this->clk_coeff = clk_coeff;
    this->min_bound = min_bound;
    this->max_bound = max_bound;
    this->learning_rate = learning_rate;
    this->initial_g2sum = initial_g2sum;
    this->initial_range = initial_range;
  }

  void set_sparse_sgd(const OptimizerConfig& optimizer_config) {
    this->nonclk_coeff = optimizer_config.nonclk_coeff;
    this->clk_coeff = optimizer_config.clk_coeff;
    this->min_bound = optimizer_config.min_bound;
    this->max_bound = optimizer_config.max_bound;
    this->learning_rate = optimizer_config.learning_rate;
    this->initial_g2sum = optimizer_config.initial_g2sum;
    this->initial_range = optimizer_config.initial_range;
  }

  void set_embedx_sgd(float mf_create_thresholds,
                      float mf_learning_rate,
                      float mf_initial_g2sum,
                      float mf_initial_range,
                      float mf_min_bound,
                      float mf_max_bound) {
    this->mf_create_thresholds = mf_create_thresholds;
    this->mf_learning_rate = mf_learning_rate;
    this->mf_initial_g2sum = mf_initial_g2sum;
    this->mf_initial_range = mf_initial_range;
    this->mf_min_bound = mf_min_bound;
    this->mf_max_bound = mf_max_bound;
  }

  void set_embedx_sgd(const OptimizerConfig& optimizer_config) {
    this->mf_create_thresholds = optimizer_config.mf_create_thresholds;
    this->mf_learning_rate = optimizer_config.mf_learning_rate;
    this->mf_initial_g2sum = optimizer_config.mf_initial_g2sum;
    this->mf_initial_range = optimizer_config.mf_initial_range;
    this->mf_min_bound = optimizer_config.mf_min_bound;
    this->mf_max_bound = optimizer_config.mf_max_bound;
  }
};

}  // namespace framework
}  // namespace paddle
