# Onnx Ugen

A SuperCollider UGen which implements the Onnx inference engine. This engine is designed for real-time safe inference of tensorflow and pytorch trained neural networks.

Installation:

Because this currently uses a dynamic library for onnx, it needs to be built from source


Building:

1. Download this repository to its own directory. Place the directory in the SC Extensions folder: "/Users/<YOUR_USER_NAME>/Library/Application Support/SuperCollider/Extensions 

    (the current mac dynamic and static libraries of onnx are included (for now))

2. Download the libsamplerate submodules:
```
git submodule update --init --recursive
```


3. Build libsamplerate in release mode (from the libsamplerate submodule directory):
(setting BUILD_TESTING to FALSE disables testing that makes it look like it didn't build correctly when it did)
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

After building make sure not to move the Onnx directory as it will need to link to the dynamic library




