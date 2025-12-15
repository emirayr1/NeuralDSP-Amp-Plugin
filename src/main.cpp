#include <intrin.h>
#include <iostream>
#include <torch/script.h>
#include "../include/AudioFile.h"

// PAGE 376

int main(){
    // Load WAV file using AudioFile
    AudioFile<float> audioFile;
    bool loadedOK = audioFile.load("../../VIOLIN 1.wav");
    
    if (!loadedOK) {
        std::cerr << "Failed to load WAV file!" << std::endl;
        return 1;
    }
    
    // Get audio data from first channel
    std::vector<float> signal = audioFile.samples[0];

    // convert float vector to tensor
    torch::Tensor in_t = torch::from_blob(signal.data(), {static_cast<int64_t>(signal.size())});

    // reshape tensor from [1, 2, 3] to [[1], [2], [3]]
    in_t = in_t.view({-1, 1});

    // convert tensor to value vector
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(in_t);
    
    torch::jit::script::Module my_lstm;
    my_lstm = torch::jit::load("../../models/my_lstm.pt");

    // feed the inputs to the network
    torch::jit::IValue out_ival = my_lstm.forward(inputs);

    // convert the return to a tuple, then extract its element
    auto out_elements = out_ival.toTuple()->elements();

    // Take the first element of the tuple and convert it to a tensor
    torch::Tensor out_t = out_elements[0].toTensor();

    // reshape the tensor from [[1], [2], [3]] to [1, 2, 3]
    out_t = out_t.view({-1});

    // convert it to a vector of floats
    float* data_ptr = out_t.data_ptr<float>();
    std::vector<float> data_vector(data_ptr, data_ptr + out_t.numel());

    // Create and configure the output audio file
    AudioFile<float> savedFile;
    savedFile.setNumChannels(1);  // mono
    savedFile.setSampleRate(44100.0);  // use same sample rate as input
    
    // Wrap data_vector in a 2D vector (one channel)
    std::vector<std::vector<float>> audioBuffer;
    audioBuffer.push_back(data_vector);
    savedFile.setAudioBuffer(audioBuffer);  // set the audio data
    
    // save it out to a file (go up to project root)
    savedFile.save("../../output.wav", AudioFileFormat::Wave);
    std::cout << "Saved " << data_vector.size() << " samples to ../../output.wav" << std::endl;

    return 0;
}