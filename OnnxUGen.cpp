// Onnx.cpp
// c++ code by Sam Pluta 2024

#include "OnnxUGen.hpp"
#include "onnxruntime_cxx_api.h"
#include "SC_PlugIn.hpp"
#include "SC_PlugIn.h"
#include <string>
#include "libsamplerate/include/samplerate.h"
#include <iostream>

static InterfaceTable *ft;


size_t OnnxUGen::resample (const float* input, float* output, size_t numSamples) noexcept
{
    SRC_DATA src_data {
        input, // data_in
        output, // data_out
        (int) numSamples, // input_frames
        int ((double) numSamples * m_ratio) + 1, // output_frames
        0, // input_frames_used
        0, // output_frames_gen
        0, // end_of_input
        m_ratio // src_ratio
    };

    src_process (src_state.get(), &src_data);
    return (size_t) src_data.output_frames_gen;
}

size_t OnnxUGen::resample_out (const float* input, float* output, size_t inSamples, size_t outSamples) noexcept
{
    SRC_DATA src_data {
        input, // data_in
        output, // data_out
        (int) inSamples, // input_frames
        (int) outSamples, // output_frames
        0, // input_frames_used
        0, // output_frames_gen
        0, // end_of_input
        1./m_ratio // src_ratio
    };

    src_process (src_state_out.get(), &src_data);
    return (size_t) src_data.output_frames_gen;
}

OnnxObject::OnnxObject() {
  //empty constructor

  std::vector<std::string> input_names(1);
  std::vector<std::string> output_names(1);
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

float OnnxObject::forward(std::vector<float>input, int num_inputs) {
// seems that we have to push the data back into the tensor...maybe we can do inference on all at once?
  std::vector<Ort::Value> input_tensors;
  input_tensors.emplace_back(vec_to_tensor<float>(input, {1, num_inputs}));

  std::vector<const char*> input_names_char(input_names.size(), nullptr);
  std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
                 [&](const std::string& str) { return str.c_str(); });

  std::vector<const char*> output_names_char(output_names.size(), nullptr);
  std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
                 [&](const std::string& str) { return str.c_str(); });

  try {

    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
                                      input_names_char.size(), output_names_char.data(), output_names_char.size());

    //should return float* instead of float
    return output_tensors[0].GetTensorMutableData<float>()[0];
  } catch (const Ort::Exception& exception) {
      std::cout << "ERROR running model inference: " << exception.what() << std::endl;
    exit(-1);
  }
}

OnnxUGen::OnnxUGen()
{

  if(in0(1)>0.f) {
      m_ratio = in0(1) / sampleRate();
  }

  m_numInputChannels = numInputs()-2;
  m_numOutputChannels = numOutputs();

  //this is the size of the output buffer for the resampling
  int in_temp_size = int(ceil(in0(1)/controlRate())*m_numInputChannels)+1;
  int out_temp_size = int(ceil(in0(1)/controlRate()))+1; //an extra one for safety

  in_rs = (float*)RTAlloc(mWorld, (double)in_temp_size * sizeof(float));
  out_temp = (float*)RTAlloc(mWorld, (double)out_temp_size * sizeof(float));

  //setting these to medium quality sample rate conversion
  //probably could be "fastest"
  int error;
  src_state.reset (src_new (SRC_SINC_MEDIUM_QUALITY, 1, &error));
  src_set_ratio (src_state.get(), m_ratio);

  int error_out;
  src_state_out.reset (src_new (SRC_SINC_MEDIUM_QUALITY, 1, &error_out));
  src_set_ratio (src_state_out.get(), 1./m_ratio);

  mCalcFunc = make_calc_function<OnnxUGen, &OnnxUGen::next>();
  next(1);
}
OnnxUGen::~OnnxUGen() {
  RTFree(mWorld, in_rs);
  RTFree(mWorld, out_temp);
}

void load_model (OnnxUGen* unit, sc_msg_iter* args) {
  const char *path = args->gets();

  // std::string pathStr = path;

  std::cout<<"Loading model from path: "<<path<<std::endl;

  try {
    std::cout<<"try"<<std::endl;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_runtime");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.SetInterOpNumThreads(1);
    // session_options.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
    // session_options.SetInterOpNumThreads(1);
    
    auto uOrtSession = std::make_unique<Ort::Session>(env, path, session_options);
    unit->m_model.session = std::move(uOrtSession);

    std::cout<<"try again"<<std::endl;

    Print("Load Onnx Model: %s\n", path);

    std::vector<std::int64_t> input_shapes;

    std::vector<std::string> input_names;

    //don't know why this is necessary
    Ort::AllocatorWithDefaultOptions allocator;

    unit->m_model_input_size = unit->m_model.session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1];
    unit->m_model_output_size = unit->m_model.session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1];

    std::cout<<"input size: "<<unit->m_model_input_size<<" output size: "<<unit->m_model_output_size<<std::endl;

    unit->m_model.input_names.emplace_back(unit->m_model.session->GetInputNameAllocated(0, allocator).get());
    std::cout << "input: " << unit->m_model.input_names.at(0) << " : " <<  std::endl;

    unit->m_model.output_names.emplace_back(unit->m_model.session->GetOutputNameAllocated(0, allocator).get());
    std::cout << "output: " << unit->m_model.output_names.at(0) << " : " <<  std::endl;

    // std::cout<<"input name: "<<unit->m_model.input_names[0]<<" output name: "<<unit->m_model.output_names[0]<<std::endl;

    int error;
    int error_out;
    unit->src_state.reset (src_new (SRC_SINC_MEDIUM_QUALITY, unit->m_numInputChannels, &error));
    unit->src_state_out.reset (src_new (SRC_SINC_MEDIUM_QUALITY, unit->m_numOutputChannels, &error_out));

    std::cout<<"input size: "<<unit->m_model_input_size<<" output size: "<<unit->m_model_output_size<<std::endl;
    std::cout<<"num input channels: "<<unit->m_numInputChannels<<" num output channels: "<<unit->m_numOutputChannels<<std::endl;

    if (unit->m_model_input_size!=unit->m_numInputChannels) {
      std::cout << "error: model input size does not match the number of input channels\n";
      std::cout << "disabling model\n";
      unit->m_model_loaded = false;
      return;
    }

    if (unit->m_model_output_size!=unit->m_numOutputChannels) {
      std::cout << "error: model output size does not match the number of output channels\n";
      std::cout << "disabling model\n";
      unit->m_model_loaded = false;
      return;
    }

    unit->m_model_loaded = true;
    //unit->m_model->reset();
    //return model;

  } catch (const std::exception& e) {
    unit->m_model_loaded = false;
    std::cerr << "error loading the model: " << e.what() << "\n";
    //return -1;
  }

}

void OnnxUGen::next(int nSamples)
{
  const float bypass = in0(0);
  const float sr = in0(1);

  const float* inArray[m_numInputChannels];
  for (int j = 0; j < m_numInputChannels; ++j) {
    inArray[j] = in(2+j);
  }

  std::vector<float> inVecSmall(m_numInputChannels);
  float outArraySmall[m_numOutputChannels];
  float *outbuf = out(0);

  if ((m_model_loaded==false)||((int)bypass==1)) {
    //Print("bypassing\n");
    for (int i = 0; i < nSamples; ++i) {
      outbuf[i] = inArray[0][i];
    }
  } else if (sr<=0||sr==sampleRate()) {
    //Print("sample rate is the same\n");
    //if the sample rate is not altered, we can just pass the data directly to the model
    for (int i = 0; i < nSamples; ++i){
      for (int j = 0; j < m_numInputChannels; ++j) {
        inVecSmall[j] = inArray[j][i];
      }
      outbuf[i] = m_model.forward(inVecSmall, m_numInputChannels);
    }
  } else {
    //if the sample rate is altered, we need to interleave and resample the input data
    float inArrayLarge[m_numInputChannels*nSamples];
    
    for (int i = 0; i < nSamples; ++i) {
      for (int j = 0; j < m_numInputChannels; ++j) {
        inArrayLarge[i*m_numInputChannels+j] = inArray[j][i];
      }
    }

    int resampled_size = resample (inArrayLarge, in_rs, nSamples);
    //if the sample rate is not altered, we can just pass the data directly to the model
    for (int i = 0; i < resampled_size; ++i){
      for (int j = 0; j < m_numInputChannels; ++j) {
        //right now it is treating the non-first channels as control rate
        inVecSmall[j] = in_rs[i*m_numInputChannels+j];
      }
      //this will only work if model is returning a single value
      out_temp[i] = m_model.forward(inVecSmall, m_numInputChannels);
    }
    int n_samps_out = resample_out (out_temp, outbuf, resampled_size, nSamples);

  }
}

PluginLoad(Onnx)
{
  ft = inTable;
  registerUnit<OnnxUGen>(ft, "Onnx", false);
  DefineUnitCmd("Onnx", "load_model", (UnitCmdFunc)&load_model);

}
