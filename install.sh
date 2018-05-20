mkdir build/
cd build/
cmake -Wno-dev -DCMAKE_BUILD_TYPE=Release ../
make
make install
cd ../
./optim
