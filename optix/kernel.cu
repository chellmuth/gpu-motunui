#include <optix.h>

#include <stdio.h>

#include "moana/driver.hpp"

using namespace moana;

extern "C" {
    __constant__ Params params;
}

extern "C" __global__ void __closesthit__ch()
{
    printf("CLOSEST HIT!\n");
}

extern "C" __global__ void __miss__ms()
{
    printf("MISS!\n");
}

extern "C" __global__ void __raygen__rg()
{
    printf("RAYGEN!\n");
}
