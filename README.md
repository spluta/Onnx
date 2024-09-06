# Onnx Ugen

A SuperCollider UGen which implements the Onnx inference engine. This engine is designed for real-time safe inference of tensorflow and pytorch trained neural networks.

The UGen can do both audio rate and control rate inference over any single-tensor-in and single-tensor-out model with any number of inputs or outputs.

Installation:

The easiest thing would be to downoad a release from the [releases](https://github.com/spluta/Onnx/releases) page.

After placing the release in the SuperCollider Extensions folder, mac users will need to dequarantine the folder. Run the following in the terminal:

xattr -cr <THE_DOWNLOADED_ONNX_FOLDER>

Building from source:

1. Clone this repository

2. Download the current static build of onnxruntime and place it in the main Onnx directory. As of this writing, it can be found here:

https://github.com/csukuangfj/onnxruntime-libs/releases

The folder will have a long name having to do with the os and cpu. Rename the folder "onnxruntime_static". 

3. Download the libsamplerate and abseil-cpp submodules:
```
git submodule update --init --recursive
```

4. Build libsamplerate in release mode (setting BUILD_TESTING to FALSE disables testing that makes it look like it didn't build correctly when it did).

from inside the libsamplerate submodule directory:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=FALSE ..
make
```

4. Build the Plugin (from the Onnx main directory):
(<PATH TO SC SOURCE> is the location of the full SuperCollider source code downloaded from github)
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSC_PATH=<PATH TO SC SOURCE> ..
cmake --build . --config Release
```

It should build Onnx plugin and leave the Onnx.scx file in the build directory






