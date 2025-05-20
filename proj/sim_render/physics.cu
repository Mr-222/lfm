#include "function/global_context.h"
#include "function/resource_manager/resource_manager.h"
#include "function/tool/fire_light_updater.h"
#include "function/type/vertex.h"
#include "nfm_init.h"
#include "nfm_util.h"
#include "physics.h"
#include <cstdio>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

void PhysicsEngineUser::initExternalMem()
{
    for (int i = 0; i < g_ctx->rm->fields.fields.size(); i++) {
        auto& field = g_ctx->rm->fields.fields[i];
#ifdef _WIN64
        HANDLE handle = g_ctx->rm->fields.getVkFieldMemHandle(i);
#else
        int fd = g_ctx->rm->fields.getVkFieldMemHandle(i);
#endif
        CudaEngine::ExtImageDesc image_desc = {
#ifdef _WIN64
            handle,
#else
            fd,
#endif
            256 * 128 * 128 * sizeof(float),
            sizeof(float),
            256,
            128,
            128,
            field.name
        };
        this->importExtImage(image_desc); // add to extBuffers internally
    }
}

void PhysicsEngineUser::init(Configuration& config, GlobalContext* g_ctx)
{

    CudaEngine::init(config, g_ctx);
    JSON_GET(DriverConfiguration, driver_cfg, config, "driver")
    total_frame     = driver_cfg.total_frame;
    frame_rate      = driver_cfg.frame_rate;
    steps_per_frame = driver_cfg.steps_per_frame;
    current_frame   = 0;
    lfm::InitNFMAsync(nfm, config.at("nfm"), streamToRun);
}

__global__ void writeToVorticity(cudaSurfaceObject_t surface_object, cudaExtent extent, size_t element_size,
                                 const float* data, int3 tile_dim, float scale)
{
    int tile_idx   = blockIdx.x;
    int3 tile_ijk  = lfm::TileIdxToIjk(tile_dim, tile_idx);
    int voxel_idx  = threadIdx.x;
    int3 voxel_ijk = lfm::VoxelIdxToIjk(voxel_idx);
    int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
    int idx        = ijk.x * tile_dim.y * tile_dim.z * 64 + ijk.y * tile_dim.z * 8 + ijk.z;
    surf3Dwrite(data[idx] * scale, surface_object, ijk.x * element_size, ijk.y, ijk.z);
}

void PhysicsEngineUser::step()
{
    waitOnSemaphore(vkUpdateSemaphore);
    lfm::GetCenteralVecAsync(*(nfm.u_), nfm.tile_dim_, *(nfm.init_u_x_), *(nfm.init_u_y_), *(nfm.init_u_z_), streamToRun);
    lfm::GetVorNormAsync(*(nfm.vor_norm_), nfm.tile_dim_, *(nfm.u_), nfm.dx_, streamToRun);
    writeToVorticity<<<32 * 16 * 16, 512, 0, streamToRun>>>(
        extImages["vorticity"].surface_object,
        extImages["vorticity"].extent,
        extImages["vorticity"].element_size,
        nfm.vor_norm_->dev_ptr_,
        { 32, 16, 16 }, 3.0f);
    signalSemaphore(cuUpdateSemaphore);

    if (total_frame < 0 || current_frame < total_frame) {
        float dt         = 1.0f / (frame_rate * nfm.reinit_every_);
        nfm.inlet_angle_ = g_ctx->rm->inlet_angle;
        for (int i = 0; i < nfm.reinit_every_; i++)
            nfm.AdvanceAsync(dt, streamToRun);
        nfm.ReinitAsync(dt, streamToRun);
        current_frame++;
    }
}

void PhysicsEngineUser::cleanup()
{
    CudaEngine::cleanup();
}
