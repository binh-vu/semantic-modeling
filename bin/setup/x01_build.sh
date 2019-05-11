cd gmtk/ctensor

rm -r build
mkdir build
cd build
cmake ..
make
sudo cp libctensor.* /usr/local/lib/