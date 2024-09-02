// Onnx.cpp
// c++ code by Sam Pluta 2024

#include "OnnxUGen.hpp"
#include "onnxruntime_cxx_api.h"
#include "SC_PlugIn.hpp"
#include "SC_PlugIn.h"
#include <string>
#include "libsamplerate/include/samplerate.h"
#include <iostream>
#include "OnnxObject.cpp"

static InterfaceTable *ft;


size_t OnnxUGen::resample (const float* input, float* output, size_t numSamples) noexcept
{
    SRC_DATA src_data {
        input, // data_in
        output, // data_out
        (int) numSamples, // input_frames
        (int) ceil ((double) numSamples * m_ratio), // output_frames
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


OnnxUGen::OnnxUGen()
{

  if(in0(1)>0.f) {
      m_ratio = in0(1) / sampleRate();
  }

  m_numInputChannels = numInputs()-2;
  m_numOutputChannels = numOutputs();

  //this is needed to handle resampling of audio when the sample rate is not the same as that at which the model was trained
  int in_size = int(ceil(sampleRate()/controlRate()))*m_numInputChannels;
  int in_rs_size = int(ceil(in0(1)/controlRate())*m_numInputChannels);
  int out_temp_size = int(ceil(in0(1)/controlRate())*m_numOutputChannels); //an extra one for safety
  int out_buf_size = int(ceil(sampleRate()/controlRate())*m_numOutputChannels); //an extra one for safety

  //std::cout<<"in_size: "<<in_size<<" in_rs_size: "<<in_rs_size<<" out_temp_size: "<<out_temp_size<<" out_buf_size: "<<out_buf_size<<std::endl;

  interleaved_array = (float*)RTAlloc(mWorld, (double)in_size * sizeof(float));
  in_rs = (float*)RTAlloc(mWorld, (double)in_rs_size * sizeof(float));
  out_temp = (float*)RTAlloc(mWorld, (double)out_temp_size * sizeof(float));
  outbuf = (float*)RTAlloc(mWorld, (double)out_buf_size * sizeof(float));

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

  std::cout<<"Loading model from path: "<<path<<std::endl;
  unit->m_model_loaded = false;
  try {
    

    unit->m_model.load_model(path);

    unit->m_model_input_size = unit->m_model.session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1];
    unit->m_model_output_size = unit->m_model.session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1];

    std::cout<<"input size: "<<unit->m_model_input_size<<" output size: "<<unit->m_model_output_size<<std::endl;

    int error;
    int error_out;
    unit->src_state.reset (src_new (SRC_SINC_MEDIUM_QUALITY, unit->m_numInputChannels, &error));
    unit->src_state_out.reset (src_new (SRC_SINC_MEDIUM_QUALITY, unit->m_numOutputChannels, &error_out));

    std::cout<<"input size: "<<unit->m_model_input_size<<" output size: "<<unit->m_model_output_size<<std::endl;
    std::cout<<"num input channels: "<<unit->m_numInputChannels<<" num output channels: "<<unit->m_numOutputChannels<<std::endl;

    unit->inVecSmall.resize(unit->m_numInputChannels);
    unit->outVecSmall.resize(unit->m_numOutputChannels);

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

  if ((m_model_loaded==false)||((int)bypass==1)) {
    for (int i = 0; i < nSamples; ++i) {
      out(0)[i] = inArray[0][i];
    }
  } else if (sr<=0||sr==sampleRate()) {
    //if the sample rate is not altered, we can just pass the data directly to the model
    for (int i = 0; i < nSamples; ++i){
      for (int j = 0; j < m_numInputChannels; ++j) {
        inVecSmall[j] = inArray[j][i];
      }
      m_model.forward(inVecSmall, outVecSmall, m_numInputChannels, m_numOutputChannels);
      for (int j = 0; j < m_numOutputChannels; ++j) {
        out(j)[i] = outVecSmall[j];
      }
    }
  } else {
    //if the model relies on a sample rate that is different than the current rate, 
    //we need to interleave and resample the input data
    //std::cout<<"resampling\n";
    for (int i = 0; i < nSamples; ++i) {
      for (int j = 0; j < m_numInputChannels; ++j) {
        interleaved_array[i*m_numInputChannels+j] = inArray[j][i];
      }
    }

    //resample the input to the model's sample rate
    int resampled_size = resample (interleaved_array, in_rs, nSamples);

    //run the model on the resampled audio
    for (int i = 0; i < resampled_size; ++i){
      for (int j = 0; j < m_numInputChannels; ++j) {
        inVecSmall[j] = in_rs[i*m_numInputChannels+j];
      }
      m_model.forward(inVecSmall, outVecSmall, m_numInputChannels, m_numOutputChannels);
      for (int j = 0; j < m_numOutputChannels; ++j) {
        out_temp[i*m_numOutputChannels+j] = outVecSmall[j];
      }
    }
    
    //resample the output back to the original sample rate
    int n_samps_out = resample_out (out_temp, outbuf, resampled_size, nSamples);

    //std::cout<<"resampled size: "<<resampled_size<<" n_samps_out: "<<n_samps_out<<std::endl;

    //deinterleave the output and put it in the output buffers
    for(int i = 0; i < n_samps_out; ++i) {
      for (int j = 0; j < m_numOutputChannels; ++j) {
        out(j)[i] = outbuf[i*m_numOutputChannels+j];
      }
    }
  }
}

PluginLoad(Onnx)
{
  ft = inTable;
  registerUnit<OnnxUGen>(ft, "Onnx", false);
  DefineUnitCmd("Onnx", "load_model", (UnitCmdFunc)&load_model);

}
