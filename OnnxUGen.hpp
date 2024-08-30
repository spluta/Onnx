
//#pragma once

#include "SC_PlugIn.hpp"
#include "libsamplerate/include/samplerate.h"
#include "onnxruntime_cxx_api.h"

// namespace Onnx {

class OnnxObject {
 public:
  OnnxObject();
  ~OnnxObject();

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  std::unique_ptr<Ort::Session> session = NULL;

  //this should be float*
  float forward (std::vector<float> input, int num_inputs);
};

class OnnxUGen : public SCUnit {
public:
  OnnxUGen();

  // Destructor
  ~OnnxUGen();

  bool m_model_loaded{false};
  // std::__1::unique_ptr<Onnx::Model<float>> m_model;
  int m_model_input_size;
  int m_model_output_size;
  int m_numInputChannels;
  int m_numOutputChannels;

  std::unique_ptr<SRC_STATE, decltype (&src_delete)> src_state { nullptr, &src_delete };
  std::unique_ptr<SRC_STATE, decltype (&src_delete)> src_state_out { nullptr, &src_delete };

  // onnxruntime setup
  //declare and empty model
  //std::unique_ptr<Ort::Session> m_model = NULL;
  OnnxObject m_model;

private:
  // Calc function
  void next(int nSamples);

  void load_model(OnnxUGen* unit, sc_msg_iter* args);

  size_t resample (const float* input, float* output, size_t numSamples) noexcept;
  size_t resample_out (const float* input, float* output, size_t inSamples, size_t outSamples) noexcept;

  float m_step_val{1.f/(float)sampleRate()};
  float m_process_sample_rate{-1.f};

  float *in_rs;
  float *out_temp;

  double m_ratio{1.0};

  int m_bypass{1};
};

// } // namespace Onnx

