#pragma once

#include "mem.h"
#include "mesh.h"
#include "setting.h"
#include "util.h"
#include <string>

namespace lfm {
template <typename T>
void ConToTileAsync(DHMemory<T>& _dst, int3 _tile_dim, const DHMemory<T>& _src, cudaStream_t _stream);

template <typename T>
void TileToConAsync(DHMemory<T>& _dst, int3 _tile_dim, const DHMemory<T>& _src, cudaStream_t _stream);

template <typename T>
void StagConToTileAsync(DHMemory<T>& _dst_x, DHMemory<T>& _dst_y, DHMemory<T>& _dst_z, int3 _tile_dim, const DHMemory<T>& _src_x, const DHMemory<T>& _src_y, const DHMemory<T>& _src_z, cudaStream_t _stream);

template <typename T>
void StagTileToConAsync(DHMemory<T>& _dst_x, DHMemory<T>& _dst_y, DHMemory<T>& _dst_z, int3 _tile_dim, const DHMemory<T>& _src_x, const DHMemory<T>& _src_y, const DHMemory<T>& _src_z, cudaStream_t _stream);

template <typename T>
void ReadNpy(std::string _file, T* _data);

template <typename T>
void WriteNpy(std::string _file, int3 _grid_dim, const T* _data);

void ReadMeshObj(std::string _file, Mesh& _meshes);

void WriteMeshObj(std::string _file, const Mesh& _meshes);
}