#include "onnxruntime_cxx_api.h"

class OnnxObject {
 public:
  OnnxObject();
  ~OnnxObject();

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  // std::vector<Ort::Value> input_tensors;
  // std::vector<Ort::Value> output_tensors;

  std::vector<const char*> input_names_char;
  std::vector<const char*> output_names_char;

  std::unique_ptr<Ort::Session> session = NULL;
  void load_model(const char* model_path);

  //this should be float*
  void forward (std::vector<float> input, std::vector<float>& output, int num_inputs, int num_outputs);
  
};