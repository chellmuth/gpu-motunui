#include "moana/driver.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>

#include <cuda_runtime.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "assert_macros.hpp"
#include "moana/renderer.hpp"
#include "scene/container.hpp"

namespace moana {

static void contextLogCallback(
    unsigned int level,
    const char *tag,
    const char *message,
    void * /*cbdata */
)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << std::endl;
}

static OptixDeviceContext createContext()
{
    // initialize CUDA
    CHECK_CUDA(cudaFree(0));

    CHECK_OPTIX(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &contextLogCallback;
    options.logCallbackLevel = 4;

    OptixDeviceContext context;
    CUcontext cuContext = 0; // current context
    CHECK_OPTIX(optixDeviceContextCreate(cuContext, &options, &context));

    return context;
}

void Driver::init()
{
    OptixDeviceContext context = createContext();

    EnvironmentLight environmentLight;
    environmentLight.queryMemoryRequirements();

    size_t gb = 1024 * 1024 * 1024;
    m_sceneState.arena.init(6.7 * gb);

    m_sceneState.environmentState = environmentLight.snapshotTextureObject(m_sceneState.arena);
    m_sceneState.arena.releaseAll();

    m_sceneState.geometries = Container::createGeometryResults(context, m_sceneState.arena);

    Pipeline::initOptixState(m_optixState, context, m_sceneState);

    CHECK_CUDA(cudaDeviceSynchronize());
}

void Driver::launch(Cam cam, const std::string &exrFilename)
{
    std::cout << "Rendering: " << exrFilename << std::endl;
    Renderer::launch(m_optixState, m_sceneState, cam, exrFilename);
}

}
