#include <iostream>
#include <oneapi/tbb.h>

#define DATA_POLICY static
#define TRANSPOSE_POLICY static

using DataPartitioner = oneapi::tbb::simple_partitioner;
static DataPartitioner dataPart;

using TransposePartitioner = oneapi::tbb::simple_partitioner;
static TransposePartitioner transPart;

using InValType = float;

#include "dataV2.hpp"

int main(int argc, char const *argv[])
{
    // pad::arrayDataV2<InValType> dataS(4, 3);
    // dataS.printA();
    // dataS.printB();

    // pad::arrayDataV2<InValType> dataOMP(11, 13, 4, "OMP");
    // dataOMP.printA();
    // dataOMP.printB();

    pad::arrayDataV2<InValType> dataTBB(11, 13, 4, "TBB", dataPart);
    dataTBB.printA();
    dataTBB.printB();

    auto iterators = dataTBB.get_range();
    auto [beginA, endA] = std::get<0>(iterators);
    auto [beginB, endB] = std::get<1>(iterators);
    auto [dataA, dataB] = dataTBB.get_ptr();

    // std::cout << "Hello World";
    return 0;
}
