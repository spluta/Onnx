// #include "OnnxObject.hpp"
#include <iostream>

OnnxObject::OnnxObject() {

}

OnnxObject::~OnnxObject() {
  if (session != NULL){
    session.release();
  }
} 

template <typename T>
Ort::Value vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape) {
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
  return tensor;
}

void OnnxObject::load_model(const char* model_path, bool stateful) {
  try {
    if (session != NULL){
      session.release();
    }

    session = NULL;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_runtime");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.SetInterOpNumThreads(1);
    session_options.SetIntraOpNumThreads(1);
    
    auto uOrtSession = std::make_unique<Ort::Session>(env, model_path, session_options);
    session = std::move(uOrtSession);

    //don't know why this is necessary
    Ort::AllocatorWithDefaultOptions allocator;

    input_names.clear();
    output_names.clear();
    input_names_char.clear();
    output_names_char.clear();

    for(int i = 0; i < session->GetInputCount(); i++) {
      input_names.emplace_back(session->GetInputNameAllocated(i, allocator).get());
      std::cout << "input: " << input_names.at(i) << " : " <<  std::endl;
    }

    for(int i = 0; i < session->GetOutputCount(); i++) {
      output_names.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
      std::cout << "output: " << output_names.at(i) << " : " <<  std::endl;
    }

    //--------------------------------

    if (stateful==1) {
      std::cout<<"Loading stateful model from path: "<<model_path<<std::endl;

      m_hidden_size = session->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape()[2];
      
      std::cout << "Hidden size: " << m_hidden_size << std::endl;
      
      //make the hidden layers that feed back into the model

      h0.assign(m_hidden_size, 0.0f);
      c0.assign(m_hidden_size, 0.0f);

      m_seq_input_shape = session->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();

      m_stateful = true;
    } else {
      std::cout<<"Loading non-stateful model from path: "<<model_path<<std::endl;
      m_stateful = false;
    }

    input_names_char.resize(input_names.size());
    std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
                  [&](const std::string& str) { return str.c_str(); });

    output_names_char.resize(output_names.size());
    std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
                  [&](const std::string& str) { return str.c_str(); });

  } catch (const Ort::Exception& exception) {
    std::cout << "ERROR loading model: " << exception.what() << std::endl;
    exit(-1);
  }
}

void OnnxObject::forward(std::vector<float> input, std::vector<float>& output, int num_inputs, int num_outputs) {
  try {

    //set the input tensors 
    input_tensors.clear();

    if(m_stateful) {
      input_tensors.emplace_back(vec_to_tensor<float>(input, {1, 1, num_inputs}));
      input_tensors.emplace_back(vec_to_tensor<float>(h0, m_seq_input_shape));
      input_tensors.emplace_back(vec_to_tensor<float>(c0, m_seq_input_shape));
    } else {
      input_tensors.emplace_back(vec_to_tensor<float>(input, {1, num_inputs}));
    }

    //do the inference
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
                                      input_names_char.size(), output_names_char.data(), output_names_char.size());

    for (int j = 0; j < output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount(); j++) {
        output[j] = output_tensors[0].GetTensorMutableData<float>()[j];
    }

    //if the model is stateful, update the hidden layers to the output of the model
    if (m_stateful) {
      //wish I knew how to do this directly with the tensors - probably more efficient
      h0.clear();
      c0.clear();

      for (int j = 0; j < output_tensors[1].GetTensorTypeAndShapeInfo().GetElementCount(); j++) {
          h0.emplace_back(output_tensors[1].GetTensorMutableData<float>()[j]);
      }

      for (int j = 0; j < output_tensors[2].GetTensorTypeAndShapeInfo().GetElementCount(); j++) {
          c0.emplace_back(output_tensors[2].GetTensorMutableData<float>()[j]);
      }
    }

  } catch (const Ort::Exception& exception) {
      std::cout << "ERROR running model inference: " << exception.what() << std::endl;
    exit(-1);
  }
}