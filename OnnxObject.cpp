// #include "OnnxObject.hpp"
#include <iostream>

OnnxObject::OnnxObject() {
  //empty constructor

    // input_names.resize(1);
    // output_names.resize(1);
    // input_tensors.resize(1);
    // output_tensors.resize(1);
    // std::vector<Ort::Value> input_tensors(1);
    // std::vector<Ort::Value> output_tensors(1);
}

OnnxObject::~OnnxObject() {
  //empty destructor
} 

template <typename T>
Ort::Value vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape) {
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
  return tensor;
}

void OnnxObject::load_model(const char* model_path) {
  try {
    std::cout<<"try"<<std::endl;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_runtime");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.SetInterOpNumThreads(1);
    session_options.SetIntraOpNumThreads(1);
    
    auto uOrtSession = std::make_unique<Ort::Session>(env, model_path, session_options);
    session = std::move(uOrtSession);

    //don't know why this is necessary
    Ort::AllocatorWithDefaultOptions allocator;

    input_names.emplace_back(session->GetInputNameAllocated(0, allocator).get());
    std::cout << "input: " << input_names.at(0) << " : " <<  std::endl;

    output_names.emplace_back(session->GetOutputNameAllocated(0, allocator).get());
    std::cout << "output: " << output_names.at(0) << " : " <<  std::endl;

    input_names_char.resize(input_names.size());
    std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
                  [&](const std::string& str) { return str.c_str(); });

    output_names_char.resize(output_names.size());
    std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
                  [&](const std::string& str) { return str.c_str(); });

    std::cout<<"try again"<<std::endl;

  } catch (const Ort::Exception& exception) {
    std::cout << "ERROR loading model: " << exception.what() << std::endl;
    exit(-1);
  }
}

void OnnxObject::forward(std::vector<float> input, std::vector<float>& output, int num_inputs, int num_outputs) {
// seems that we have to push the data back into the tensor...maybe we can do inference on all at once?
  std::vector<Ort::Value> input_tensors;
  input_tensors.emplace_back(vec_to_tensor<float>(input, {1, num_inputs}));
  //input_tensors[0] = vec_to_tensor<float>(input, {1, num_inputs});

  try {

    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
                                      input_names_char.size(), output_names_char.data(), output_names_char.size());

    for (int j = 0; j < output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount(); j++) {
        output[j] = output_tensors[0].GetTensorMutableData<float>()[j];
        
    }
    // std::cout << std::endl;

  } catch (const Ort::Exception& exception) {
      std::cout << "ERROR running model inference: " << exception.what() << std::endl;
    exit(-1);
  }
}