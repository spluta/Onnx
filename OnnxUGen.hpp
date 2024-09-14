#include "SC_PlugIn.hpp"
#include "libsamplerate/include/samplerate.h"
#include "OnnxObject.hpp"


class OnnxUGen : public SCUnit {
public:
  OnnxUGen();

  // Destructor
  ~OnnxUGen();

  bool m_model_loaded{false};
  int m_model_input_size;
  int m_model_output_size;
  int m_numInputChannels;
  int m_numOutputChannels;

  std::vector<float> inVecSmall;
  std::vector<float> outVecSmall;

  std::vector<std::vector<float>> outVecs;

  float* interleaved_array;
  float* outbuf;

  std::unique_ptr<SRC_STATE, decltype (&src_delete)> src_state { nullptr, &src_delete };
  std::unique_ptr<SRC_STATE, decltype (&src_delete)> src_state_out { nullptr, &src_delete };

  // onnxruntime setup
  //declare and empty model
  //std::unique_ptr<Ort::Session> m_model = NULL;
  OnnxObject m_model;

private:
  // Calc function
  void next(int nSamples);

  // Load model
  void load_model(OnnxUGen* unit, sc_msg_iter* args);

  void load_stateful_rnn (OnnxUGen* unit, sc_msg_iter* args, int stateful);

  size_t resample (const float* input, float* output, size_t numSamples) noexcept;
  size_t resample_out (const float* input, float* output, size_t inSamples, size_t outSamples) noexcept;

  float m_step_val{1.f/(float)sampleRate()};
  float m_process_sample_rate{-1.f};
  bool m_stateful{false};

  float *in_rs;
  float *out_temp;

  double m_ratio{1.0};

  int m_bypass{1};
};

// } // namespace Onnx

