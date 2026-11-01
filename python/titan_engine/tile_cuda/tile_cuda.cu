#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace titan_cuda_detail {

constexpr int kThreads = 256;
constexpr int kMaxExpandPerSplat = 4096;
constexpr int kMaxTilesPerSegment = 4096;

int div_ceil(int a, int b) { return (a + b - 1) / b; }

void check_cuda(const char* message) {
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, message, ": ", cudaGetErrorString(err));
}

__global__ void project_kernel(
    const float* __restrict__ positions,
    const float* __restrict__ scales,
    const float* __restrict__ rotation,
    const float* __restrict__ origin,
    float fx,
    float fy,
    float cx,
    float cy,
    float near,
    float far,
    int width,
    int height,
    int tile_size,
    int tiles_x,
    int tiles_y,
    int max_expand,
    int n,
    int* __restrict__ active_flags,
    float* __restrict__ proj_u,
    float* __restrict__ proj_v,
    float* __restrict__ proj_sigma_u,
    float* __restrict__ proj_sigma_v,
    float* __restrict__ proj_z,
    int* __restrict__ min_tile_u,
    int* __restrict__ max_tile_u,
    int* __restrict__ min_tile_v,
    int* __restrict__ max_tile_v,
    int* __restrict__ tile_counts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float px = positions[i * 3 + 0];
  float py = positions[i * 3 + 1];
  float pz = positions[i * 3 + 2];
  float ox = origin[0];
  float oy = origin[1];
  float oz = origin[2];
  float rx = px - ox;
  float ry = py - oy;
  float rz = pz - oz;
  // camera_space = rel @ rotation (row-vector convention).
  float cam_x =
      rx * rotation[0] + ry * rotation[3] + rz * rotation[6];
  float cam_y =
      rx * rotation[1] + ry * rotation[4] + rz * rotation[7];
  float cam_z =
      rx * rotation[2] + ry * rotation[5] + rz * rotation[8];
  active_flags[i] = 0;
  min_tile_u[i] = 0;
  max_tile_u[i] = 0;
  min_tile_v[i] = 0;
  max_tile_v[i] = 0;
  tile_counts[i] = 0;
  if (!(cam_z > near && cam_z < far)) return;
  float scale = scales[i];
  float z = cam_z;
  float u = fx * cam_x / z + cx;
  float v = fy * cam_y / z + cy;
  float sigma_u = scale * fx / z;
  float sigma_v = scale * fy / z;
  float radius_u = 2.8f * sigma_u;
  float radius_v = 2.8f * sigma_v;
  const float margin = 4.0f;
  if (u <= -radius_u - margin || u >= width + radius_u + margin) return;
  if (v <= -radius_v - margin || v >= height + radius_v + margin) return;
  proj_u[i] = u;
  proj_v[i] = v;
  proj_sigma_u[i] = sigma_u;
  proj_sigma_v[i] = sigma_v;
  proj_z[i] = z;
  int lo_u = (int)floorf((u - radius_u - margin) / (float)tile_size);
  int hi_u = (int)floorf((u + radius_u + margin) / (float)tile_size);
  int lo_v = (int)floorf((v - radius_v - margin) / (float)tile_size);
  int hi_v = (int)floorf((v + radius_v + margin) / (float)tile_size);
  lo_u = std::max(0, lo_u);
  hi_u = std::min(tiles_x - 1, hi_u);
  lo_v = std::max(0, lo_v);
  hi_v = std::min(tiles_y - 1, hi_v);
  int span_u = hi_u - lo_u + 1;
  int span_v = hi_v - lo_v + 1;
  int count = std::min(span_u * span_v, max_expand);
  active_flags[i] = 1;
  min_tile_u[i] = lo_u;
  max_tile_u[i] = hi_u;
  min_tile_v[i] = lo_v;
  max_tile_v[i] = hi_v;
  tile_counts[i] = count;
}

__global__ void compact_projection_kernel(
    const int* __restrict__ active_flags,
    const int* __restrict__ prefix,
    const float* __restrict__ proj_u,
    const float* __restrict__ proj_v,
    const float* __restrict__ proj_sigma_u,
    const float* __restrict__ proj_sigma_v,
    const float* __restrict__ proj_z,
    const int* __restrict__ min_tile_u,
    const int* __restrict__ max_tile_u,
    const int* __restrict__ min_tile_v,
    const int* __restrict__ max_tile_v,
    const int* __restrict__ tile_counts,
    int n,
    int* __restrict__ active_ids,
    float* __restrict__ compact_u,
    float* __restrict__ compact_v,
    float* __restrict__ compact_sigma_u,
    float* __restrict__ compact_sigma_v,
    float* __restrict__ compact_z,
    int* __restrict__ compact_min_u,
    int* __restrict__ compact_max_u,
    int* __restrict__ compact_min_v,
    int* __restrict__ compact_max_v,
    int* __restrict__ compact_tile_counts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || active_flags[i] == 0) return;
  int out = prefix[i];
  active_ids[out] = i;
  compact_u[out] = proj_u[i];
  compact_v[out] = proj_v[i];
  compact_sigma_u[out] = proj_sigma_u[i];
  compact_sigma_v[out] = proj_sigma_v[i];
  compact_z[out] = proj_z[i];
  compact_min_u[out] = min_tile_u[i];
  compact_max_u[out] = max_tile_u[i];
  compact_min_v[out] = min_tile_v[i];
  compact_max_v[out] = max_tile_v[i];
  compact_tile_counts[out] = tile_counts[i];
}

__global__ void expand_pairs_per_splat_kernel(
    const int* __restrict__ active_flags,
    const float* __restrict__ proj_z,
    const int* __restrict__ min_tile_u,
    const int* __restrict__ max_tile_u,
    const int* __restrict__ min_tile_v,
    const int* __restrict__ max_tile_v,
    const int* __restrict__ tile_counts,
    const int64_t* __restrict__ pair_offsets,
    int64_t pair_cap,
    int tiles_x,
    int n,
    int64_t* __restrict__ pair_keys,
    int* __restrict__ pair_values) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n || active_flags[idx] == 0) return;
  int count = tile_counts[idx];
  if (count == 0) return;
  int64_t base = pair_offsets[idx];
  if (base >= pair_cap) return;
  if ((int64_t)count > pair_cap - base) {
    count = static_cast<int>(pair_cap - base);
  }
  int span_u = max_tile_u[idx] - min_tile_u[idx] + 1;
  int lo_u = min_tile_u[idx];
  int lo_v = min_tile_v[idx];
  float z = proj_z[idx];
  uint32_t depth_key = __float_as_uint(z);
  for (int k = 0; k < count; ++k) {
    int tu = lo_u + k % span_u;
    int tv = lo_v + k / span_u;
    int tile = tv * tiles_x + tu;
    pair_keys[base + k] = ((int64_t)tile << 32) | (int64_t)depth_key;
    pair_values[base + k] = idx;
  }
}

__global__ void expand_pairs_per_splat_kernel_legacy(
    const float* __restrict__ compact_z,
    const int* __restrict__ compact_min_u,
    const int* __restrict__ compact_max_u,
    const int* __restrict__ compact_min_v,
    const int* __restrict__ compact_max_v,
    const int* __restrict__ compact_tile_counts,
    const int64_t* __restrict__ pair_offsets,
    int tiles_x,
    int active_count,
    int64_t* __restrict__ pair_keys,
    int* __restrict__ pair_values) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= active_count) return;
  int count = compact_tile_counts[idx];
  if (count == 0) return;
  int base = pair_offsets[idx];
  int span_u = compact_max_u[idx] - compact_min_u[idx] + 1;
  int lo_u = compact_min_u[idx];
  int lo_v = compact_min_v[idx];
  float z = compact_z[idx];
  uint32_t depth_key = __float_as_uint(z);
  for (int k = 0; k < count; ++k) {
    int tu = lo_u + k % span_u;
    int tv = lo_v + k / span_u;
    int tile = tv * tiles_x + tu;
    pair_keys[base + k] = ((int64_t)tile << 32) | (int64_t)depth_key;
    pair_values[base + k] = idx;
  }
}

__global__ void fill_pair_sentinels_kernel(
    int64_t* __restrict__ pair_keys,
    int* __restrict__ pair_values,
    int total_cap,
    int sentinel_tile) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= total_cap) return;
  pair_keys[j] = ((int64_t)sentinel_tile << 32) | 0xFFFFFFFFLL;
  pair_values[j] = 0;
}

__global__ void tile_pair_start_kernel(
    const int64_t* __restrict__ pair_keys,
    int total_pairs,
    int tile_count,
    int* __restrict__ tile_pair_start) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= total_pairs) return;
  int tile = (int)(pair_keys[j] >> 32);
  if (tile >= tile_count) return;
  if (j == 0 || tile != (int)(pair_keys[j - 1] >> 32)) {
    tile_pair_start[tile] = j;
  }
}

__global__ void tile_pair_count_kernel(
    const int64_t* __restrict__ pair_keys,
    int total_pairs,
    const int* __restrict__ tile_pair_start,
    int tile_count,
    int* __restrict__ tile_pair_count) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= total_pairs) return;
  int tile = (int)(pair_keys[j] >> 32);
  if (tile >= tile_count) return;
  if (j == total_pairs - 1 ||
      tile != (int)(pair_keys[j + 1] >> 32)) {
    int start = tile_pair_start[tile];
    tile_pair_count[tile] = j - start + 1;
  }
}

__global__ void truncate_tile_pair_count_kernel(
    int* __restrict__ tile_pair_count,
    int tile_count,
    int tile_max_splats) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= tile_count) return;
  if (tile_pair_count[t] > tile_max_splats) {
    tile_pair_count[t] = tile_max_splats;
  }
}

__global__ void pixel_tile_keys_kernel(
    const float* __restrict__ pixels,
    int pixel_count,
    int tile_size,
    int tiles_x,
    int tiles_y,
    int* __restrict__ pixel_keys,
    int64_t* __restrict__ pixel_values) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pixel_count) return;
  int tu = (int)floorf(pixels[p * 2 + 0] / (float)tile_size);
  int tv = (int)floorf(pixels[p * 2 + 1] / (float)tile_size);
  tu = std::max(0, std::min(tiles_x - 1, tu));
  tv = std::max(0, std::min(tiles_y - 1, tv));
  pixel_keys[p] = tv * tiles_x + tu;
  pixel_values[p] = (int64_t)p;
}

__global__ void gather_sorted_pixels_kernel(
    const float* __restrict__ pixels,
    const int64_t* __restrict__ pixel_order,
    int pixel_count,
    float* __restrict__ sorted_pixels) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pixel_count) return;
  int64_t src = pixel_order[p];
  sorted_pixels[p * 2 + 0] = pixels[src * 2 + 0];
  sorted_pixels[p * 2 + 1] = pixels[src * 2 + 1];
}

__global__ void tile_pixel_start_kernel(
    const int* __restrict__ pixel_keys,
    int pixel_count,
    int* __restrict__ tile_pixel_start) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pixel_count) return;
  int tile = pixel_keys[p];
  if (p == 0 || tile != pixel_keys[p - 1]) {
    tile_pixel_start[tile] = p;
  }
}

__global__ void tile_pixel_count_kernel(
    const int* __restrict__ pixel_keys,
    int pixel_count,
    const int* __restrict__ tile_pixel_start,
    int* __restrict__ tile_pixel_count) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pixel_count) return;
  int tile = pixel_keys[p];
  if (p == pixel_count - 1 || tile != pixel_keys[p + 1]) {
    int start = tile_pixel_start[tile];
    tile_pixel_count[tile] = p - start + 1;
  }
}

__global__ void tile_pixel_end_kernel(
    const int* __restrict__ pixel_keys,
    int pixel_count,
    int* __restrict__ tile_pixel_end) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pixel_count) return;
  int tile = pixel_keys[p];
  if (p == pixel_count - 1 ||
      tile != pixel_keys[p + 1]) {
    tile_pixel_end[tile] = p + 1;
  }
}

__global__ void segment_tile_data_kernel(
    const int* __restrict__ global_tile_pair_start,
    const int* __restrict__ global_tile_pair_count,
    int t0,
    int t1,
    int pair_base,
    int* __restrict__ seg_tile_pair_start,
    int* __restrict__ seg_tile_pair_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= t1 - t0) return;
  int t = t0 + idx;
  int local = idx;
  seg_tile_pair_start[local] =
      global_tile_pair_start[t] - pair_base;
  seg_tile_pair_count[local] = global_tile_pair_count[t];
}

__global__ void fill_background_range_kernel(
    float* __restrict__ output,
    int p_start,
    int p_end,
    const float* __restrict__ background) {
  int p = blockIdx.x * blockDim.x + threadIdx.x + p_start;
  if (p >= p_end) return;
  output[p * 3 + 0] = background[0];
  output[p * 3 + 1] = background[1];
  output[p * 3 + 2] = background[2];
}

__global__ void fill_background_kernel(
    const float* __restrict__ background,
    int pixel_count,
    float* __restrict__ output) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pixel_count) return;
  output[p * 3 + 0] = background[0];
  output[p * 3 + 1] = background[1];
  output[p * 3 + 2] = background[2];
}

__global__ void pixel_work_count_kernel(
    const float* __restrict__ sorted_pixels,
    const int* __restrict__ tile_pair_count,
    int tile_size,
    int tiles_x,
    int tiles_y,
    int pixel_count,
    int64_t* __restrict__ pixel_work) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pixel_count) return;
  int tu = (int)floorf(sorted_pixels[p * 2 + 0] / (float)tile_size);
  int tv = (int)floorf(sorted_pixels[p * 2 + 1] / (float)tile_size);
  tu = std::max(0, std::min(tiles_x - 1, tu));
  tv = std::max(0, std::min(tiles_y - 1, tv));
  int tile = tv * tiles_x + tu;
  pixel_work[p] = (int64_t)tile_pair_count[tile];
}

__global__ void forward_kernel(
    const float* __restrict__ sorted_pixels,
    const int* __restrict__ tile_pair_start,
    const int* __restrict__ tile_pair_count,
    const int* __restrict__ sorted_splats,
    const int* __restrict__ active_ids,
    const float* __restrict__ proj_u,
    const float* __restrict__ proj_v,
    const float* __restrict__ proj_sigma_u,
    const float* __restrict__ proj_sigma_v,
    const float* __restrict__ opacity,
    const float* __restrict__ colors,
    const float* __restrict__ background,
    int tile_size,
    int tiles_x,
    int tiles_y,
    float disk_cutoff,
    float falloff_power,
    int tile_start,
    int pair_base,
    int p_start,
    int p_end,
    const int64_t* __restrict__ pixel_work_offset,
    int64_t work_base,
    float* __restrict__ seg_trans,
    float* __restrict__ output) {
  int p = blockIdx.x * blockDim.x + threadIdx.x + p_start;
  if (p >= p_end) return;
  float px = sorted_pixels[p * 2 + 0];
  float py = sorted_pixels[p * 2 + 1];
  int tu = (int)floorf(px / (float)tile_size);
  int tv = (int)floorf(py / (float)tile_size);
  tu = std::max(0, std::min(tiles_x - 1, tu));
  tv = std::max(0, std::min(tiles_y - 1, tv));
  int tile = tv * tiles_x + tu;
  int local_tile = tile - tile_start;
  int s0 = tile_pair_start[local_tile];
  int s1 = s0 + tile_pair_count[local_tile];
  int64_t wbase = pixel_work_offset[p] - work_base;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
  float trans = 1.0f;
  float cutoff = 0.5f * disk_cutoff * disk_cutoff;
  for (int s = s0; s < s1; ++s) {
    int ai = sorted_splats[pair_base + s];
    int gid = active_ids[ai];
    float u = proj_u[ai];
    float v = proj_v[ai];
    float su = proj_sigma_u[ai];
    float sv = proj_sigma_v[ai];
    float du = px - u;
    float dv = py - v;
    float sq = du * du / (2.0f * su * su) +
               dv * dv / (2.0f * sv * sv);
    float weight = 0.0f;
    if (sq < cutoff) {
      if (falloff_power == 1.0f) {
        weight = __expf(-sq);
      } else {
        weight = __expf(-powf(sq, falloff_power));
      }
    }
    float raw = opacity[gid] * weight;
    float alpha = fminf(fmaxf(raw, 0.0f), 1.0f);
    seg_trans[wbase + (int64_t)(s - s0)] = trans;
    float survive = 1.0f - alpha;
    acc0 += alpha * trans * colors[gid * 3 + 0];
    acc1 += alpha * trans * colors[gid * 3 + 1];
    acc2 += alpha * trans * colors[gid * 3 + 2];
    trans *= survive;
  }
  output[p * 3 + 0] = acc0 + trans * background[0];
  output[p * 3 + 1] = acc1 + trans * background[1];
  output[p * 3 + 2] = acc2 + trans * background[2];
}

__global__ void transmittance_kernel(
    const float* __restrict__ sorted_pixels,
    const int* __restrict__ tile_pair_start,
    const int* __restrict__ tile_pair_count,
    const int* __restrict__ sorted_splats,
    const int* __restrict__ active_ids,
    const float* __restrict__ proj_u,
    const float* __restrict__ proj_v,
    const float* __restrict__ proj_sigma_u,
    const float* __restrict__ proj_sigma_v,
    const float* __restrict__ opacity,
    int tile_size,
    int tiles_x,
    int tiles_y,
    float disk_cutoff,
    float falloff_power,
    int tile_start,
    int pair_base,
    int p_start,
    int p_end,
    const int64_t* __restrict__ pixel_work_offset,
    int64_t work_base,
    float* __restrict__ seg_trans) {
  int p = blockIdx.x * blockDim.x + threadIdx.x + p_start;
  if (p >= p_end) return;
  float px = sorted_pixels[p * 2 + 0];
  float py = sorted_pixels[p * 2 + 1];
  int tu = (int)floorf(px / (float)tile_size);
  int tv = (int)floorf(py / (float)tile_size);
  tu = std::max(0, std::min(tiles_x - 1, tu));
  tv = std::max(0, std::min(tiles_y - 1, tv));
  int tile = tv * tiles_x + tu;
  int local_tile = tile - tile_start;
  int s0 = tile_pair_start[local_tile];
  int s1 = s0 + tile_pair_count[local_tile];
  int64_t wbase = pixel_work_offset[p] - work_base;
  float trans = 1.0f;
  float cutoff = 0.5f * disk_cutoff * disk_cutoff;
  for (int s = s0; s < s1; ++s) {
    int ai = sorted_splats[pair_base + s];
    int gid = active_ids[ai];
    float u = proj_u[ai];
    float v = proj_v[ai];
    float su = proj_sigma_u[ai];
    float sv = proj_sigma_v[ai];
    float du = px - u;
    float dv = py - v;
    float sq = du * du / (2.0f * su * su) +
               dv * dv / (2.0f * sv * sv);
    float weight = 0.0f;
    if (sq < cutoff) {
      if (falloff_power == 1.0f) {
        weight = __expf(-sq);
      } else {
        weight = __expf(-powf(sq, falloff_power));
      }
    }
    float raw = opacity[gid] * weight;
    float alpha = fminf(fmaxf(raw, 0.0f), 1.0f);
    seg_trans[wbase + (int64_t)(s - s0)] = trans;
    trans *= (1.0f - alpha);
  }
}

__global__ void backward_kernel(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ pixel_order,
    const float* __restrict__ sorted_pixels,
    const int* __restrict__ tile_pair_start,
    const int* __restrict__ tile_pair_count,
    const int* __restrict__ sorted_splats,
    const int* __restrict__ active_ids,
    const float* __restrict__ proj_u,
    const float* __restrict__ proj_v,
    const float* __restrict__ proj_sigma_u,
    const float* __restrict__ proj_sigma_v,
    const float* __restrict__ proj_z,
    const int64_t* __restrict__ pixel_work_offset,
    int64_t work_base,
    const float* __restrict__ seg_trans,
    const float* __restrict__ opacity,
    const float* __restrict__ colors,
    const float* __restrict__ background,
    const float* __restrict__ rotation,
    float fx,
    float fy,
    float cx,
    float cy,
    int tile_size,
    int tiles_x,
    int tiles_y,
    float disk_cutoff,
    float falloff_power,
    int tile_start,
    int pair_base,
    int p_start,
    int p_end,
    float* __restrict__ grad_positions,
    float* __restrict__ grad_scale,
    float* __restrict__ grad_opacity,
    float* __restrict__ grad_colors) {
  int p = blockIdx.x * blockDim.x + threadIdx.x + p_start;
  if (p >= p_end) return;
  float px = sorted_pixels[p * 2 + 0];
  float py = sorted_pixels[p * 2 + 1];
  int tu = (int)floorf(px / (float)tile_size);
  int tv = (int)floorf(py / (float)tile_size);
  tu = std::max(0, std::min(tiles_x - 1, tu));
  tv = std::max(0, std::min(tiles_y - 1, tv));
  int tile = tv * tiles_x + tu;
  int64_t ogid = pixel_order[p];
  float g0 = grad_output[ogid * 3 + 0];
  float g1 = grad_output[ogid * 3 + 1];
  float g2 = grad_output[ogid * 3 + 2];
  float gbg = g0 * background[0] + g1 * background[1] +
              g2 * background[2];
  int local_tile = tile - tile_start;
  int s0 = tile_pair_start[local_tile];
  int s1 = s0 + tile_pair_count[local_tile];
  int64_t wbase = pixel_work_offset[p] - work_base;
  float cutoff = 0.5f * disk_cutoff * disk_cutoff;
  float R = gbg;
  for (int s = s1 - 1; s >= s0; --s) {
    int ai = sorted_splats[pair_base + s];
    int gid = active_ids[ai];
    float u = proj_u[ai];
    float v = proj_v[ai];
    float su = proj_sigma_u[ai];
    float sv = proj_sigma_v[ai];
    float du = px - u;
    float dv = py - v;
    float sq = du * du / (2.0f * su * su) +
               dv * dv / (2.0f * sv * sv);
    float weight = 0.0f;
    if (sq < cutoff) {
      if (falloff_power == 1.0f) {
        weight = __expf(-sq);
      } else {
        weight = __expf(-powf(sq, falloff_power));
      }
    }
    float raw = opacity[gid] * weight;
    float alpha = fminf(fmaxf(raw, 0.0f), 1.0f);
    float c0 = colors[gid * 3 + 0];
    float c1 = colors[gid * 3 + 1];
    float c2 = colors[gid * 3 + 2];
    float T = seg_trans[wbase + (int64_t)(s - s0)];
    float gdotc = g0 * c0 + g1 * c1 + g2 * c2;
    float R_prev = alpha * gdotc + (1.0f - alpha) * R;
    float dalpha = T * (gdotc - R);
    atomicAdd(&grad_colors[gid * 3 + 0], alpha * T * g0);
    atomicAdd(&grad_colors[gid * 3 + 1], alpha * T * g1);
    atomicAdd(&grad_colors[gid * 3 + 2], alpha * T * g2);
    if (raw > 0.0f && raw < 1.0f) {
      float dopacity = dalpha * weight;
      atomicAdd(&grad_opacity[gid], dopacity);
      float dweight = dalpha * opacity[gid];
      float dsq = 0.0f;
      if (weight > 0.0f) {
        float dw_dsq;
        if (falloff_power == 1.0f) {
          dw_dsq = -weight;
        } else {
          dw_dsq = -falloff_power * powf(sq, falloff_power - 1.0f) * weight;
        }
        dsq = dweight * dw_dsq;
      }
      float dL_du = -dsq * du / (su * su);
      float dL_dv = -dsq * dv / (sv * sv);
      float dL_dsu = dsq * (-du * du / (su * su * su));
      float dL_dsv = dsq * (-dv * dv / (sv * sv * sv));
      float z = proj_z[ai];
      float scale_val = su * z / fx;
      float dL_dz = dL_dsu * (-scale_val * fx / (z * z)) +
                    dL_dsv * (-scale_val * fy / (z * z));
      float x = (u - cx) * z / fx;
      float y = (v - cy) * z / fy;
      dL_dz += dL_du * (-fx * x / (z * z)) +
               dL_dv * (-fy * y / (z * z));
      float dL_dx = dL_du * fx / z;
      float dL_dy = dL_dv * fy / z;
      atomicAdd(&grad_scale[gid], dL_dsu * (fx / z) + dL_dsv * (fy / z));
      atomicAdd(&grad_positions[gid * 3 + 0],
                dL_dx * rotation[0] + dL_dy * rotation[1] +
                dL_dz * rotation[2]);
      atomicAdd(&grad_positions[gid * 3 + 1],
                dL_dx * rotation[3] + dL_dy * rotation[4] +
                dL_dz * rotation[5]);
      atomicAdd(&grad_positions[gid * 3 + 2],
                dL_dx * rotation[6] + dL_dy * rotation[7] +
                dL_dz * rotation[8]);
    }
    R = R_prev;
  }
}

__global__ void max_tile_pair_count_kernel(
    const int* __restrict__ tile_pair_count,
    int tile_count,
    int* __restrict__ max_count) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= tile_count) return;
  int v = tile_pair_count[t];
  if (v > 0) atomicMax(max_count, v);
}

__global__ void total_work_kernel(
    const int64_t* __restrict__ pixel_work_offset,
    const int* __restrict__ tile_pair_count,
    const float* __restrict__ sorted_pixels,
    int pixel_count,
    int tile_size,
    int tiles_x,
    int tiles_y,
    int64_t* __restrict__ total_work) {
  if (pixel_count <= 0) {
    total_work[0] = 0;
    return;
  }
  int p = pixel_count - 1;
  int tu = (int)floorf(sorted_pixels[p * 2 + 0] / (float)tile_size);
  int tv = (int)floorf(sorted_pixels[p * 2 + 1] / (float)tile_size);
  tu = std::max(0, std::min(tiles_x - 1, tu));
  tv = std::max(0, std::min(tiles_y - 1, tv));
  int tile = tv * tiles_x + tu;
  total_work[0] = pixel_work_offset[p] + (int64_t)tile_pair_count[tile];
}

__global__ void forward_kernel_batched(
    const float* __restrict__ sorted_pixels,
    const int* __restrict__ tile_pair_start,
    const int* __restrict__ tile_pair_count,
    const int* __restrict__ sorted_splats,
    const float* __restrict__ proj_u,
    const float* __restrict__ proj_v,
    const float* __restrict__ proj_sigma_u,
    const float* __restrict__ proj_sigma_v,
    const float* __restrict__ opacity,
    const float* __restrict__ colors,
    const float* __restrict__ background,
    int tile_size,
    int tiles_x,
    int tiles_y,
    float disk_cutoff,
    float falloff_power,
    int pixel_count,
    float* __restrict__ output) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pixel_count) return;
  float px = sorted_pixels[p * 2 + 0];
  float py = sorted_pixels[p * 2 + 1];
  int tu = (int)floorf(px / (float)tile_size);
  int tv = (int)floorf(py / (float)tile_size);
  tu = std::max(0, std::min(tiles_x - 1, tu));
  tv = std::max(0, std::min(tiles_y - 1, tv));
  int tile = tv * tiles_x + tu;
  int s0 = tile_pair_start[tile];
  int s1 = s0 + tile_pair_count[tile];
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
  float trans = 1.0f;
  float cutoff = 0.5f * disk_cutoff * disk_cutoff;
  for (int s = s0; s < s1; ++s) {
    int gid = sorted_splats[s];
    float u = proj_u[gid];
    float v = proj_v[gid];
    float su = proj_sigma_u[gid];
    float sv = proj_sigma_v[gid];
    float du = px - u;
    float dv = py - v;
    float sq = du * du / (2.0f * su * su) +
               dv * dv / (2.0f * sv * sv);
    float weight = 0.0f;
    if (sq < cutoff) {
      if (falloff_power == 1.0f) {
        weight = __expf(-sq);
      } else {
        weight = __expf(-powf(sq, falloff_power));
      }
    }
    float raw = opacity[gid] * weight;
    float alpha = fminf(fmaxf(raw, 0.0f), 1.0f);
    acc0 += alpha * trans * colors[gid * 3 + 0];
    acc1 += alpha * trans * colors[gid * 3 + 1];
    acc2 += alpha * trans * colors[gid * 3 + 2];
    trans *= 1.0f - alpha;
  }
  output[p * 3 + 0] = acc0 + trans * background[0];
  output[p * 3 + 1] = acc1 + trans * background[1];
  output[p * 3 + 2] = acc2 + trans * background[2];
}

__global__ void backward_kernel_batched(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ pixel_order,
    const float* __restrict__ sorted_pixels,
    const int* __restrict__ tile_pair_start,
    const int* __restrict__ tile_pair_count,
    const int* __restrict__ sorted_splats,
    const float* __restrict__ proj_u,
    const float* __restrict__ proj_v,
    const float* __restrict__ proj_sigma_u,
    const float* __restrict__ proj_sigma_v,
    const float* __restrict__ proj_z,
    const int64_t* __restrict__ pixel_work_offset,
    int use_compact,
    int scratch_stride,
    float* __restrict__ trans_scratch,
    const float* __restrict__ opacity,
    const float* __restrict__ colors,
    const float* __restrict__ background,
    const float* __restrict__ rotation,
    float fx,
    float fy,
    float cx,
    float cy,
    int tile_size,
    int tiles_x,
    int tiles_y,
    float disk_cutoff,
    float falloff_power,
    int p_start,
    int p_end,
    float* __restrict__ grad_positions,
    float* __restrict__ grad_scale,
    float* __restrict__ grad_opacity,
    float* __restrict__ grad_colors) {
  int p = blockIdx.x * blockDim.x + threadIdx.x + p_start;
  if (p >= p_end) return;
  float px = sorted_pixels[p * 2 + 0];
  float py = sorted_pixels[p * 2 + 1];
  int tu = (int)floorf(px / (float)tile_size);
  int tv = (int)floorf(py / (float)tile_size);
  tu = std::max(0, std::min(tiles_x - 1, tu));
  tv = std::max(0, std::min(tiles_y - 1, tv));
  int tile = tv * tiles_x + tu;
  int s0 = tile_pair_start[tile];
  int s1 = s0 + tile_pair_count[tile];
  int64_t wbase = use_compact
                      ? pixel_work_offset[p]
                      : (int64_t)(p - p_start) * scratch_stride;
  float cutoff = 0.5f * disk_cutoff * disk_cutoff;
  float trans = 1.0f;
  for (int s = s0; s < s1; ++s) {
    int gid = sorted_splats[s];
    float u = proj_u[gid];
    float v = proj_v[gid];
    float su = proj_sigma_u[gid];
    float sv = proj_sigma_v[gid];
    float du = px - u;
    float dv = py - v;
    float sq = du * du / (2.0f * su * su) +
               dv * dv / (2.0f * sv * sv);
    float weight = 0.0f;
    if (sq < cutoff) {
      if (falloff_power == 1.0f) {
        weight = __expf(-sq);
      } else {
        weight = __expf(-powf(sq, falloff_power));
      }
    }
    float raw = opacity[gid] * weight;
    float alpha = fminf(fmaxf(raw, 0.0f), 1.0f);
    trans_scratch[wbase + (int64_t)(s - s0)] = trans;
    trans *= 1.0f - alpha;
  }
  int64_t ogid = pixel_order[p];
  float g0 = grad_output[ogid * 3 + 0];
  float g1 = grad_output[ogid * 3 + 1];
  float g2 = grad_output[ogid * 3 + 2];
  float gbg = g0 * background[0] + g1 * background[1] +
              g2 * background[2];
  float R = gbg;
  for (int s = s1 - 1; s >= s0; --s) {
    int gid = sorted_splats[s];
    float u = proj_u[gid];
    float v = proj_v[gid];
    float su = proj_sigma_u[gid];
    float sv = proj_sigma_v[gid];
    float du = px - u;
    float dv = py - v;
    float sq = du * du / (2.0f * su * su) +
               dv * dv / (2.0f * sv * sv);
    float weight = 0.0f;
    if (sq < cutoff) {
      if (falloff_power == 1.0f) {
        weight = __expf(-sq);
      } else {
        weight = __expf(-powf(sq, falloff_power));
      }
    }
    float raw = opacity[gid] * weight;
    float alpha = fminf(fmaxf(raw, 0.0f), 1.0f);
    float c0 = colors[gid * 3 + 0];
    float c1 = colors[gid * 3 + 1];
    float c2 = colors[gid * 3 + 2];
    float T = trans_scratch[wbase + (int64_t)(s - s0)];
    float gdotc = g0 * c0 + g1 * c1 + g2 * c2;
    float R_prev = alpha * gdotc + (1.0f - alpha) * R;
    float dalpha = T * (gdotc - R);
    atomicAdd(&grad_colors[gid * 3 + 0], alpha * T * g0);
    atomicAdd(&grad_colors[gid * 3 + 1], alpha * T * g1);
    atomicAdd(&grad_colors[gid * 3 + 2], alpha * T * g2);
    if (raw > 0.0f && raw < 1.0f) {
      float dopacity = dalpha * weight;
      atomicAdd(&grad_opacity[gid], dopacity);
      float dweight = dalpha * opacity[gid];
      float dsq = 0.0f;
      if (weight > 0.0f) {
        float dw_dsq;
        if (falloff_power == 1.0f) {
          dw_dsq = -weight;
        } else {
          dw_dsq = -falloff_power * powf(sq, falloff_power - 1.0f) * weight;
        }
        dsq = dweight * dw_dsq;
      }
      float dL_du = -dsq * du / (su * su);
      float dL_dv = -dsq * dv / (sv * sv);
      float dL_dsu = dsq * (-du * du / (su * su * su));
      float dL_dsv = dsq * (-dv * dv / (sv * sv * sv));
      float z = proj_z[gid];
      float scale_val = su * z / fx;
      float dL_dz = dL_dsu * (-scale_val * fx / (z * z)) +
                    dL_dsv * (-scale_val * fy / (z * z));
      float x = (u - cx) * z / fx;
      float y = (v - cy) * z / fy;
      dL_dz += dL_du * (-fx * x / (z * z)) +
               dL_dv * (-fy * y / (z * z));
      float dL_dx = dL_du * fx / z;
      float dL_dy = dL_dv * fy / z;
      atomicAdd(&grad_scale[gid], dL_dsu * (fx / z) + dL_dsv * (fy / z));
      atomicAdd(&grad_positions[gid * 3 + 0],
                dL_dx * rotation[0] + dL_dy * rotation[1] +
                dL_dz * rotation[2]);
      atomicAdd(&grad_positions[gid * 3 + 1],
                dL_dx * rotation[3] + dL_dy * rotation[4] +
                dL_dz * rotation[5]);
      atomicAdd(&grad_positions[gid * 3 + 2],
                dL_dx * rotation[6] + dL_dy * rotation[7] +
                dL_dz * rotation[8]);
    }
    R = R_prev;
  }
}

// host render helpers
bool _is_cuda_float(const torch::Tensor& t) {
  return t.is_cuda() && t.scalar_type() == torch::kFloat32 &&
         t.is_contiguous();
}
bool _is_cuda_int(const torch::Tensor& t) {
  return t.is_cuda() && t.scalar_type() == torch::kInt32 &&
         t.is_contiguous();
}
bool _is_cuda_long(const torch::Tensor& t) {
  return t.is_cuda() && t.scalar_type() == torch::kInt64 &&
         t.is_contiguous();
}

std::vector<torch::Tensor> render_forward_gpu(
    torch::Tensor positions,
    torch::Tensor scales,
    torch::Tensor opacity,
    torch::Tensor colors,
    torch::Tensor rotation,
    torch::Tensor origin,
    torch::Tensor pixels,
    torch::Tensor background,
    double fx,
    double fy,
    double cx,
    double cy,
    int64_t width,
    int64_t height,
    int64_t tile_size,
    double near,
    double far,
    double disk_cutoff,
    double falloff_power,
    int64_t tile_pair_budget,
    int64_t tile_max_splats,
    int64_t tile_work_budget) {
  TORCH_CHECK(_is_cuda_float(positions), "positions must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_float(scales), "scales must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_float(opacity), "opacity must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_float(colors), "colors must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_float(rotation), "rotation must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_float(origin), "origin must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_float(pixels), "pixels must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_float(background), "background must be contiguous CUDA float32");
  const int n = static_cast<int>(positions.size(0));
  const int pixel_count = static_cast<int>(pixels.size(0));
  const int tsize = static_cast<int>(tile_size);
  const int tiles_x = static_cast<int>(div_ceil(static_cast<int>(width), tsize));
  const int tiles_y = static_cast<int>(div_ceil(static_cast<int>(height), tsize));
  const int tile_count = tiles_x * tiles_y;
  const int64_t pair_cap = std::max<int64_t>(tile_pair_budget, 1);
  const int tile_max = static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>(tile_max_splats, 1 << 20)));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto policy = thrust::cuda::par.on(stream);

  auto opt_f = torch::TensorOptions().dtype(torch::kFloat32).device(positions.device());
  auto opt_i = torch::TensorOptions().dtype(torch::kInt32).device(positions.device());
  auto opt_l = torch::TensorOptions().dtype(torch::kInt64).device(positions.device());

  auto active_flags = torch::zeros({n}, opt_i);
  auto proj_u = torch::empty({n}, opt_f);
  auto proj_v = torch::empty({n}, opt_f);
  auto proj_sigma_u = torch::empty({n}, opt_f);
  auto proj_sigma_v = torch::empty({n}, opt_f);
  auto proj_z = torch::empty({n}, opt_f);
  auto min_tile_u = torch::empty({n}, opt_i);
  auto max_tile_u = torch::empty({n}, opt_i);
  auto min_tile_v = torch::empty({n}, opt_i);
  auto max_tile_v = torch::empty({n}, opt_i);
  auto tile_counts = torch::empty({n}, opt_i);

  const int max_expand = std::min<int>(
      kMaxExpandPerSplat,
      static_cast<int>(std::max<int64_t>(
          pair_cap / std::max<int64_t>(n, 1), 1)));
  project_kernel<<<div_ceil(n, kThreads), kThreads, 0, stream>>>(
      positions.data_ptr<float>(),
      scales.data_ptr<float>(),
      rotation.data_ptr<float>(),
      origin.data_ptr<float>(),
      static_cast<float>(fx),
      static_cast<float>(fy),
      static_cast<float>(cx),
      static_cast<float>(cy),
      static_cast<float>(near),
      static_cast<float>(far),
      static_cast<int>(width),
      static_cast<int>(height),
      tsize,
      tiles_x,
      tiles_y,
      max_expand,
      n,
      active_flags.data_ptr<int>(),
      proj_u.data_ptr<float>(),
      proj_v.data_ptr<float>(),
      proj_sigma_u.data_ptr<float>(),
      proj_sigma_v.data_ptr<float>(),
      proj_z.data_ptr<float>(),
      min_tile_u.data_ptr<int>(),
      max_tile_u.data_ptr<int>(),
      min_tile_v.data_ptr<int>(),
      max_tile_v.data_ptr<int>(),
      tile_counts.data_ptr<int>());
  check_cuda("project_kernel launch failed");

  auto output = torch::empty({pixel_count, 3}, opt_f);
  fill_background_kernel<<<div_ceil(pixel_count, kThreads), kThreads, 0, stream>>>(
      background.data_ptr<float>(), pixel_count, output.data_ptr<float>());
  check_cuda("fill_background_kernel launch failed");

  auto pair_keys = torch::empty({pair_cap}, opt_l);
  auto pair_values = torch::empty({pair_cap}, opt_i);
  fill_pair_sentinels_kernel<<<div_ceil((int)pair_cap, kThreads), kThreads, 0, stream>>>(
      pair_keys.data_ptr<int64_t>(),
      pair_values.data_ptr<int>(),
      static_cast<int>(pair_cap),
      tile_count);
  check_cuda("fill_pair_sentinels_kernel launch failed");

  auto pair_offsets = torch::empty({n + 1}, opt_l);
  thrust::exclusive_scan(
      policy,
      thrust::device_pointer_cast(tile_counts.data_ptr<int>()),
      thrust::device_pointer_cast(tile_counts.data_ptr<int>()) + n,
      thrust::device_pointer_cast(pair_offsets.data_ptr<int64_t>()));
  check_cuda("pair exclusive_scan failed");

  expand_pairs_per_splat_kernel<<<div_ceil(n, kThreads), kThreads, 0, stream>>>(
      active_flags.data_ptr<int>(),
      proj_z.data_ptr<float>(),
      min_tile_u.data_ptr<int>(),
      max_tile_u.data_ptr<int>(),
      min_tile_v.data_ptr<int>(),
      max_tile_v.data_ptr<int>(),
      tile_counts.data_ptr<int>(),
      pair_offsets.data_ptr<int64_t>(),
      pair_cap,
      tiles_x,
      n,
      pair_keys.data_ptr<int64_t>(),
      pair_values.data_ptr<int>());
  check_cuda("expand_pairs_per_splat_kernel launch failed");

  thrust::sort_by_key(
      policy,
      thrust::device_pointer_cast(pair_keys.data_ptr<int64_t>()),
      thrust::device_pointer_cast(pair_keys.data_ptr<int64_t>()) + pair_cap,
      thrust::device_pointer_cast(pair_values.data_ptr<int>()));
  check_cuda("pair sort_by_key failed");

  auto tile_pair_start = torch::zeros({tile_count}, opt_i);
  auto tile_pair_count = torch::zeros({tile_count}, opt_i);
  tile_pair_start_kernel<<<div_ceil((int)pair_cap, kThreads), kThreads, 0, stream>>>(
      pair_keys.data_ptr<int64_t>(),
      static_cast<int>(pair_cap),
      tile_count,
      tile_pair_start.data_ptr<int>());
  tile_pair_count_kernel<<<div_ceil((int)pair_cap, kThreads), kThreads, 0, stream>>>(
      pair_keys.data_ptr<int64_t>(),
      static_cast<int>(pair_cap),
      tile_pair_start.data_ptr<int>(),
      tile_count,
      tile_pair_count.data_ptr<int>());
  truncate_tile_pair_count_kernel<<<div_ceil(tile_count, kThreads), kThreads, 0, stream>>>(
      tile_pair_count.data_ptr<int>(),
      tile_count,
      tile_max);
  check_cuda("tile pair metadata kernels failed");

  auto pixel_keys = torch::empty({pixel_count}, opt_i);
  auto pixel_order = torch::empty({pixel_count}, opt_l);
  auto sorted_pixels = torch::empty({pixel_count, 2}, opt_f);
  pixel_tile_keys_kernel<<<div_ceil(pixel_count, kThreads), kThreads, 0, stream>>>(
      pixels.data_ptr<float>(),
      pixel_count,
      tsize,
      tiles_x,
      tiles_y,
      pixel_keys.data_ptr<int>(),
      pixel_order.data_ptr<int64_t>());
  thrust::sort_by_key(
      policy,
      thrust::device_pointer_cast(pixel_keys.data_ptr<int>()),
      thrust::device_pointer_cast(pixel_keys.data_ptr<int>()) + pixel_count,
      thrust::device_pointer_cast(pixel_order.data_ptr<int64_t>()));
  gather_sorted_pixels_kernel<<<div_ceil(pixel_count, kThreads), kThreads, 0, stream>>>(
      pixels.data_ptr<float>(),
      pixel_order.data_ptr<int64_t>(),
      pixel_count,
      sorted_pixels.data_ptr<float>());
  check_cuda("pixel sort/gather failed");

  auto tile_pixel_start = torch::zeros({tile_count}, opt_i);
  auto tile_pixel_count = torch::zeros({tile_count}, opt_i);
  auto tile_pixel_end = torch::zeros({tile_count}, opt_i);
  tile_pixel_start_kernel<<<div_ceil(pixel_count, kThreads), kThreads, 0, stream>>>(
      pixel_keys.data_ptr<int>(),
      pixel_count,
      tile_pixel_start.data_ptr<int>());
  tile_pixel_count_kernel<<<div_ceil(pixel_count, kThreads), kThreads, 0, stream>>>(
      pixel_keys.data_ptr<int>(),
      pixel_count,
      tile_pixel_start.data_ptr<int>(),
      tile_pixel_count.data_ptr<int>());
  tile_pixel_end_kernel<<<div_ceil(pixel_count, kThreads), kThreads, 0, stream>>>(
      pixel_keys.data_ptr<int>(),
      pixel_count,
      tile_pixel_end.data_ptr<int>());
  check_cuda("tile pixel metadata kernels failed");

  auto pixel_work = torch::empty({pixel_count}, opt_l);
  auto pixel_work_offset = torch::empty({pixel_count}, opt_l);
  pixel_work_count_kernel<<<div_ceil(pixel_count, kThreads), kThreads, 0, stream>>>(
      sorted_pixels.data_ptr<float>(),
      tile_pair_count.data_ptr<int>(),
      tsize,
      tiles_x,
      tiles_y,
      pixel_count,
      pixel_work.data_ptr<int64_t>());
  thrust::exclusive_scan(
      policy,
      thrust::device_pointer_cast(pixel_work.data_ptr<int64_t>()),
      thrust::device_pointer_cast(pixel_work.data_ptr<int64_t>()) + pixel_count,
      thrust::device_pointer_cast(pixel_work_offset.data_ptr<int64_t>()));
  check_cuda("pixel work scan failed");

  forward_kernel_batched<<<div_ceil(pixel_count, kThreads), kThreads, 0, stream>>>(
      sorted_pixels.data_ptr<float>(),
      tile_pair_start.data_ptr<int>(),
      tile_pair_count.data_ptr<int>(),
      pair_values.data_ptr<int>(),
      proj_u.data_ptr<float>(),
      proj_v.data_ptr<float>(),
      proj_sigma_u.data_ptr<float>(),
      proj_sigma_v.data_ptr<float>(),
      opacity.data_ptr<float>(),
      colors.data_ptr<float>(),
      background.data_ptr<float>(),
      tsize,
      tiles_x,
      tiles_y,
      static_cast<float>(disk_cutoff),
      static_cast<float>(falloff_power),
      pixel_count,
      output.data_ptr<float>());
  check_cuda("forward_kernel_batched launch failed");

  std::vector<torch::Tensor> result;
  result.reserve(17);
  result.push_back(output);
  result.push_back(pixel_order);
  result.push_back(sorted_pixels);
  result.push_back(tile_pixel_start);
  result.push_back(tile_pixel_count);
  result.push_back(tile_pair_start);
  result.push_back(tile_pair_count);
  result.push_back(pair_values);
  result.push_back(torch::empty({0}, opt_i));
  result.push_back(proj_u);
  result.push_back(proj_v);
  result.push_back(proj_sigma_u);
  result.push_back(proj_sigma_v);
  result.push_back(proj_z);
  result.push_back(pixel_work_offset);
  result.push_back(torch::zeros({0, 6}, opt_l));
  const int64_t seg_work_cap = std::max<int64_t>(1, tile_work_budget);
  result.push_back(torch::empty({std::max<int64_t>(seg_work_cap, tile_max)}, opt_f));
  return result;
}

std::vector<torch::Tensor> render_backward(
    torch::Tensor grad_output,
    torch::Tensor pixel_order,
    torch::Tensor sorted_pixels,
    torch::Tensor tile_pair_start,
    torch::Tensor tile_pair_count,
    torch::Tensor sorted_splats,
    torch::Tensor active_ids,
    torch::Tensor compact_u,
    torch::Tensor compact_v,
    torch::Tensor compact_sigma_u,
    torch::Tensor compact_sigma_v,
    torch::Tensor compact_z,
    torch::Tensor pixel_work_offset,
    torch::Tensor opacity,
    torch::Tensor colors,
    torch::Tensor background,
    torch::Tensor rotation,
    double fx,
    double fy,
    double cx,
    double cy,
    int64_t tile_size,
    int64_t tiles_x,
    int64_t tiles_y,
    double disk_cutoff,
    double falloff_power,
    torch::Tensor segments,
    torch::Tensor seg_trans) {
  TORCH_CHECK(_is_cuda_float(grad_output), "grad_output must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_long(pixel_order), "pixel_order must be contiguous CUDA int64");
  TORCH_CHECK(_is_cuda_float(sorted_pixels), "sorted_pixels must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_int(tile_pair_start), "tile_pair_start must be contiguous CUDA int32");
  TORCH_CHECK(_is_cuda_int(tile_pair_count), "tile_pair_count must be contiguous CUDA int32");
  TORCH_CHECK(_is_cuda_int(sorted_splats), "sorted_splats must be contiguous CUDA int32");
  TORCH_CHECK(_is_cuda_int(active_ids), "active_ids must be contiguous CUDA int32");
  TORCH_CHECK(_is_cuda_float(compact_u), "compact_u must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_float(compact_v), "compact_v must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_float(compact_sigma_u), "compact_sigma_u must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_float(compact_sigma_v), "compact_sigma_v must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_float(compact_z), "compact_z must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_long(pixel_work_offset), "pixel_work_offset must be contiguous CUDA int64");
  TORCH_CHECK(_is_cuda_float(opacity), "opacity must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_float(colors), "colors must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_float(background), "background must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_float(rotation), "rotation must be contiguous CUDA float32");
  TORCH_CHECK(_is_cuda_long(segments), "segments must be contiguous CUDA int64");
  TORCH_CHECK(_is_cuda_float(seg_trans), "seg_trans must be contiguous CUDA float32");

  const int n = static_cast<int>(opacity.size(0));
  const int pixel_count = static_cast<int>(pixel_order.size(0));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto opt_f = torch::TensorOptions().dtype(torch::kFloat32).device(opacity.device());
  auto opt_i = torch::TensorOptions().dtype(torch::kInt32).device(opacity.device());

  auto grad_positions = torch::zeros({n, 3}, opt_f);
  auto grad_scale = torch::zeros({n}, opt_f);
  auto grad_opacity = torch::zeros({n}, opt_f);
  auto grad_colors = torch::zeros({n, 3}, opt_f);
  if (n == 0 || pixel_count == 0) {
    return {grad_positions, grad_scale, grad_opacity, grad_colors};
  }

  const int tile_count = static_cast<int>(tiles_x * tiles_y);
  auto tile_max = torch::zeros({1}, opt_i);
  max_tile_pair_count_kernel<<<div_ceil(tile_count, kThreads), kThreads, 0, stream>>>(
      tile_pair_count.data_ptr<int>(),
      tile_count,
      tile_max.data_ptr<int>());
  check_cuda("max_tile_pair_count_kernel failed");
  const int tile_max_pairs = tile_max.cpu().data_ptr<int>()[0];
  if (tile_max_pairs <= 0) {
    return {grad_positions, grad_scale, grad_opacity, grad_colors};
  }

  // Chunked backward: recompute transmittance per pixel block on GPU only.
  const int64_t seg_work_cap = std::max<int64_t>(
      1, std::min<int64_t>(seg_trans.numel(), (int64_t)1 << 30));
  const int chunk_pixels = static_cast<int>(
      std::max<int64_t>(1, seg_work_cap / tile_max_pairs));

  for (int p_start = 0; p_start < pixel_count; p_start += chunk_pixels) {
    const int p_end = std::min(pixel_count, p_start + chunk_pixels);
    backward_kernel_batched<<<div_ceil(p_end - p_start, kThreads), kThreads, 0, stream>>>(
        grad_output.data_ptr<float>(),
        pixel_order.data_ptr<int64_t>(),
        sorted_pixels.data_ptr<float>(),
        tile_pair_start.data_ptr<int>(),
        tile_pair_count.data_ptr<int>(),
        sorted_splats.data_ptr<int>(),
        compact_u.data_ptr<float>(),
        compact_v.data_ptr<float>(),
        compact_sigma_u.data_ptr<float>(),
        compact_sigma_v.data_ptr<float>(),
        compact_z.data_ptr<float>(),
        pixel_work_offset.data_ptr<int64_t>(),
        0,
        tile_max_pairs,
        seg_trans.data_ptr<float>(),
        opacity.data_ptr<float>(),
        colors.data_ptr<float>(),
        background.data_ptr<float>(),
        rotation.data_ptr<float>(),
        static_cast<float>(fx),
        static_cast<float>(fy),
        static_cast<float>(cx),
        static_cast<float>(cy),
        static_cast<int>(tile_size),
        static_cast<int>(tiles_x),
        static_cast<int>(tiles_y),
        static_cast<float>(disk_cutoff),
        static_cast<float>(falloff_power),
        p_start,
        p_end,
        grad_positions.data_ptr<float>(),
        grad_scale.data_ptr<float>(),
        grad_opacity.data_ptr<float>(),
        grad_colors.data_ptr<float>());
  }
  check_cuda("backward_kernel_batched failed");
  return {grad_positions, grad_scale, grad_opacity, grad_colors};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render_forward", &render_forward_gpu, "render forward");
  m.def("render_backward", &render_backward, "render backward");
}

}  // namespace titan_cuda_detail

