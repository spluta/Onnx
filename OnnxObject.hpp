#include "onnxruntime_cxx_api.h"

class OnnxObject {
 public:
  OnnxObject();
  ~OnnxObject();

  std::vector<Ort::Value> input_tensors;

  std::vector<float> h0;
  std::vector<float> c0;

  std::vector<Ort::Value> hidden_layers;

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  std::vector<const char*> input_names_char;
  std::vector<const char*> output_names_char;

  std::unique_ptr<Ort::Session> session = NULL;
  void load_model(const char* model_path, bool stateful);

  void forward (std::vector<float> input, std::vector<float>& output, int num_inputs, int num_outputs);

  bool m_stateful{false};
  int m_hidden_size;
  std::vector<int64_t> m_seq_input_shape;
  
};