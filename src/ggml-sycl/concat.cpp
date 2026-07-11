//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "ggml.h"

#include "concat.hpp"
#include "siriuth.hpp"

#define SYCL_CONCAT_WORK_GROUP_NUM 99999
#define SYCL_CONCAT_WORK_GROUP_SIZE 32 // S8 85.68s S16 78.93s S32 79.49s
//#define SYCL_CONCAT_WORK_GROUP_SIZE 64 // S16 85.33s  S32 80.97s
//#define SYCL_CONCAT_SUB_GROUP_SIZE 8
#define SYCL_CONCAT_SUB_GROUP_SIZE 16
//#define SYCL_CONCAT_SUB_GROUP_SIZE 32

static inline size_t elem_size(ggml_type t) {
    return ggml_type_size(t) / ggml_blck_size(t);
}
/*
template <typename T>
static void concat_T_dim0(const T *x, const T *y, T *dst,
                            const int ne0, const int ne00,
                            const sycl::nd_item<3> &item_ct1) {
    int nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (nidx >= ne0) {
        return;
    }
    // operation
    int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
    if (nidx < ne00) { // src0
        int offset_src = nidx + item_ct1.get_group(1) * ne00 +
                     item_ct1.get_group(0) * ne00 * item_ct1.get_group_range(1);
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx - ne00 + item_ct1.get_group(1) * (ne0 - ne00) +
            item_ct1.get_group(0) * (ne0 - ne00) * item_ct1.get_group_range(1);
        dst[offset_dst] = y[offset_src];
    }
}
*/

template <typename T>
static void concat_T_dim0_offset(const T *src0, const T *src1, T *dst,
    const int ne0, const int ne1, const int ne2,
    const int ne00, const int ne01,
    const sycl::range<3> offset,
    const sycl::nd_item<3> &item_ct1)
{
    const int i0 = item_ct1.get_global_id(2) + offset[2];
    const int i1 = item_ct1.get_global_id(1) + offset[1];
    const int i2 = item_ct1.get_global_id(0) + offset[0];

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2) {
        return;
    }
    // operation
    dst       [i0      + i1 *  ne0        + i2 *  ne0        * ne1 ] = (i0 < ne00)
        ? src0[i0      + i1 *  ne00       + i2 *  ne00       * ne01]
        : src1[i0-ne00 + i1 * (ne0 -ne00) + i2 * (ne0 -ne00) * ne01];
}

/*
template <typename T>
static void concat_T_dim1(const T *x, const T *y, T *dst,
                            const int ne0, const int ne01,
                            const sycl::nd_item<3> &item_ct1) {
    int nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (nidx >= ne0) {
        return;
    }
    // operation
    int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
    if (item_ct1.get_group(1) < (size_t) ne01) { // src0
        int offset_src =
            nidx + item_ct1.get_group(1) * ne0 + item_ct1.get_group(0) * ne0 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx + (item_ct1.get_group(1) - ne01) * ne0 +
            item_ct1.get_group(0) * ne0 * (item_ct1.get_group_range(1) - ne01);
        dst[offset_dst] = y[offset_src];
    }
}
*/

template <typename T>
static void concat_T_dim1_offset(const T *src0, const T *src1, T *dst,
    const int ne0, const int ne1, const int ne2,
    //const int ne00,
    const int ne01,
    const sycl::range<3> offset,
    const sycl::nd_item<3> &item_ct1)
{
    const int i0 = item_ct1.get_global_id(2) + offset[2];
    const int i1 = item_ct1.get_global_id(1) + offset[1];
    const int i2 = item_ct1.get_global_id(0) + offset[0];

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2) {
        return;
    }
    // operation
    dst       [i0 +  i1       * ne0 + i2 * ne0 *  ne1      ] = (i1 < ne01)
        ? src0[i0 +  i1       * ne0 + i2 * ne0 *  ne01     ]
        : src1[i0 + (i1-ne01) * ne0 + i2 * ne0 * (ne1-ne01)];
}

/*
template <typename T>
static void concat_T_dim2(const T *x, const T *y, T *dst,
                            const int ne0, const int ne02,
                            const sycl::nd_item<3> &item_ct1) {
    int nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (nidx >= ne0) {
      return;
    }
    // operation
    int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
    if (item_ct1.get_group(0) < (size_t) ne02) { // src0
        int offset_src = nidx + item_ct1.get_group(1) * ne0 +
                     item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx + item_ct1.get_group(1) * ne0 +
            (item_ct1.get_group(0) - ne02) * ne0 * item_ct1.get_group_range(1);
        dst[offset_dst] = y[offset_src];
    }
}
*/

template <typename T>
static void concat_T_dim2_offset(const T *src0, const T *src1, T *dst,
    const int ne0, const int ne1, const int ne2,
    const int ne02,
    const sycl::range<3> offset,
    const sycl::nd_item<3> &item_ct1)
{
    const int i0 = item_ct1.get_global_id(2) + offset[2];
    const int i1 = item_ct1.get_global_id(1) + offset[1];
    const int i2 = item_ct1.get_global_id(0) + offset[0];

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2) {
      return;
    }

    // operation
    dst       [i0 + i1 * ne0 +  i2       * ne0 * ne1] = (i2 < ne02)
        ? src0[i0 + i1 * ne0 +  i2       * ne0 * ne1]
        : src1[i0 + i1 * ne0 + (i2-ne02) * ne0 * ne1];
}

/*
template <typename T>
static void concat_T_sycl(const T *x, const T *y, T *dst,
                            int ne00, int ne01, int ne02, int ne0, int ne1,
                            int ne2, int dim, queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s dim:%d\n", __func__, dim);
    int num_blocks = (ne0 + SYCL_CONCAT_BLOCK_SIZE - 1) / SYCL_CONCAT_BLOCK_SIZE;
    sycl::range<3> gridDim(ne2, ne1, num_blocks);
    switch (dim) {
    case 0:
        stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim0<T>(x, y, dst, ne0, ne00, item_ct1); });
        break;
    case 1:
        stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim1<T>(x, y, dst, ne0, ne01, item_ct1); });
        break;
    // dim >=2 will be dispatched to the default path
    default:
        stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim2<T>(x, y, dst, ne0, ne02, item_ct1); });
        break;
    }
}
*/

template <typename T>
static void concat_T_sycl(const T *x, const T *y, T *dst,
                            int ne00, int ne01, int ne02, int ne0, int ne1,
                            int ne2, int dim, queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s dim:%d\n", __func__, dim);
    int num_blocks = (ne0 + SYCL_CONCAT_BLOCK_SIZE - 1) / SYCL_CONCAT_BLOCK_SIZE;
    sycl::range<3> gridDim(ne2, ne1, num_blocks);
    sycl::range<3> world(ne2, ne1, ne0);
    sycl::range<3> local(1, 1, SYCL_CONCAT_WORK_GROUP_SIZE);
    switch (dim) {
    case 0:
        //stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
        //                                  sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
        //                [=](sycl::nd_item<3> item_ct1) { concat_T_dim0<T>(x, y, dst, ne0, ne00, item_ct1); });

        ggml_sycl_looper(world, local, SYCL_CONCAT_WORK_GROUP_NUM, stream,
            [=](sycl::range<3> global, sycl::range<3> offset){

                auto e =
                stream->parallel_for(
                    sycl::nd_range<3>(global, local),
                    [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(SYCL_CONCAT_SUB_GROUP_SIZE)]]
                    {
                        concat_T_dim0_offset<T>(x, y, dst,
                            ne0, ne1, ne2,
                            ne00, ne01,
                            offset,
                           item_ct1);
                    }
                );
                SyclQueueEventWatcher::getInstance().SetEvent(e);

            }
        );

        break;
    case 1:
/*
        stream->parallel_for(
            sycl::nd_range<3>(
                gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)
            ),
            [=](sycl::nd_item<3> item_ct1) {
                concat_T_dim1<T>(x, y, dst, ne0, ne01, item_ct1);
            }
        );
*/

        ggml_sycl_looper(world, local, SYCL_CONCAT_WORK_GROUP_NUM, stream,
            [=](sycl::range<3> global, sycl::range<3> offset){

                auto e =
                stream->parallel_for(
                    sycl::nd_range<3>(global, local),
                    [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(SYCL_CONCAT_SUB_GROUP_SIZE)]]
                    {
                        concat_T_dim1_offset<T>(x, y, dst,
                            ne0, ne1, ne2,
                            //ne00,
                            ne01,
                            offset,
                            item_ct1);
                    }
                );
                SyclQueueEventWatcher::getInstance().SetEvent(e);


            }
        );

        break;
    // dim >=2 will be dispatched to the default path
    default:
/*
        stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim2<T>(x, y, dst, ne0, ne02, item_ct1); });
*/
        ggml_sycl_looper(world, local, SYCL_CONCAT_WORK_GROUP_NUM, stream,
            [=](sycl::range<3> global, sycl::range<3> offset){

                auto e =
                stream->parallel_for(
                    sycl::nd_range<3>(global, local),
                    [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(SYCL_CONCAT_SUB_GROUP_SIZE)]]
                    {
                        concat_T_dim2_offset<T>(x, y, dst,
                            ne0, ne1, ne2,
                            ne02,
                            offset, item_ct1);
                    }
                );
                SyclQueueEventWatcher::getInstance().SetEvent(e);


            }
        );

        break;
    }
}

// non-contiguous kernel (slow)
/*
template<typename T>
static void concat_T_sycl_non_cont(
    queue_ptr stream, const char *src0, const char *src1, char *dst,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    uint64_t nb00, uint64_t nb01, uint64_t nb02, uint64_t nb03,
    //int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
    int64_t, int64_t, int64_t, int64_t,
    uint64_t nb10,
    uint64_t nb11, uint64_t nb12, uint64_t nb13, int64_t ne0, int64_t ne1,
    int64_t ne2, int64_t ne3, uint64_t nb0, uint64_t nb1, uint64_t nb2,
    uint64_t nb3, int32_t dim) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
  sycl::range<3> gridDim(ne3, ne2, ne1);
  stream->parallel_for(sycl::nd_range<3>(gridDim, sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
      int64_t i3 = item_ct1.get_group(0);
      int64_t i2 = item_ct1.get_group(1);
      int64_t i1 = item_ct1.get_group(2);

      int64_t o[4] = { 0, 0, 0, 0 };
      o[dim]       = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

      const T * x;

      for (int i0 = item_ct1.get_local_id(2); i0 < ne0; i0 += item_ct1.get_local_range(2)) {
          if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
              x = (const T *) (src0 + (i3) *nb03 + (i2) *nb02 + (i1) *nb01 + (i0) *nb00);
          } else {
              x = (const T *) (src1 + (i3 - o[3]) * nb13 + (i2 - o[2]) * nb12 + (i1 - o[1]) * nb11 +
                                   (i0 - o[0]) * nb10);
          }

          T *y = (T *)(dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

          *y = *x;
      }
  });
}
*/
/*
template<typename T>
//static __dpct_inline__ void concat_T_non_cont_offset( // inline展開は意味がない可能性…
static void concat_T_non_cont_offset(
    //const char *src0, const char *src1, char *dst,
    ////const T *x, const T *y, T *dst,
    //int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    //size_t  nb00, size_t  nb01, size_t  nb02, size_t  nb03,
    ////int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
    //size_t  nb10, size_t  nb11, size_t  nb12, size_t  nb13,
    //int64_t ne0,  int64_t ne1,  int64_t ne2,  int64_t ne3,
    //size_t  nb0,  size_t  nb1,  size_t  nb2,  size_t  nb3,

    const char *src0, const char *src1, char *dst,
    size_t ne00, size_t ne01, size_t ne02, size_t ne03,
    size_t  nb00, size_t  nb01, size_t  nb02, size_t  nb03,
    //int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
    size_t  nb10, size_t  nb11, size_t  nb12, size_t  nb13,
    size_t ne0,  size_t ne1,  size_t ne2,  size_t ne3,
    size_t  nb0,  size_t  nb1,  size_t  nb2,  size_t  nb3,
    int32_t dim,
    const sycl::range<3> offset,
    const sycl::nd_item<3> &item_ct1)
{
    //int64_t i3 = item_ct1.get_group(0);
    //int64_t i2 = item_ct1.get_group(1);
    //int64_t i1 = item_ct1.get_group(2);
    //const int64_t i0 = item_ct1.get_global_id(2) + offset[2];
    //const int64_t i1 = item_ct1.get_global_id(1) + offset[1];
    //const int64_t i2s= item_ct1.get_global_id(0) + offset[0];
    //const int64_t i2 = i2s / ne3;
    const size_t i0 = item_ct1.get_global_id(2) + offset[2];
    const size_t i1 = item_ct1.get_global_id(1) + offset[1];
    const size_t i2s= item_ct1.get_global_id(0) + offset[0];
    const size_t i2 = i2s / ne3;
    if (i0 >= ne0 || i1 >= ne1 || i2s >= ne2*ne3 || i2 >= ne2) {
        return;
    }
    //const int64_t i3 = i2s % ne3;
    const size_t i3 = i2s % ne3;


    const T *x;
    if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
        x = (const T *) (src0 + i3*nb03 +  i2*nb02 +  i1*nb01 +  i0*nb00);
    } else {
        //// これは微妙な前提だが…
        //x = (const T *) src1 + (i3-ne03)*nb13 +  (i2-ne02)*nb12 +  (i1-ne01)*nb11 +  (i0-ne00)*nb10;
        x = (const T *) (src1 + (i3-(dim==3?ne03:0))*nb13 +  (i2-(dim==2?ne02:0))*nb12 +  (i1-(dim==1?ne01:0))*nb11 +  (i0-(dim==0?ne00:0))*nb10);
        //int64_t o[4] = { 0, 0, 0, 0 };
        //o[dim]       = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));
        //x = (const T *) (src1 + (i3 - o[3]) * nb13 + (i2 - o[2]) * nb12 + (i1 - o[1]) * nb11 + (i0 - o[0]) * nb10);
    }
    //T *y = (T *)(dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);
    // *y = *x;
    (T *)(dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0) = *x;
}
*/

/*
// このロジックが大元となっている。
// できればこのまま残しておくことをお勧め。
template<typename T>
static void concat_T_non_cont_offset(
// Flux2Klein 4B で 1Step 78.93s total 125.29s 早くはなった。
// static void concat_T_non_cont_offset( // 1step 142.61s total 189.12s
    const char *src0, const char *src1, char *dst,
    const size_t *ne0,
    const size_t *nb0,
    const size_t *nb1,
    const size_t *ne,
    const size_t *nb,
    const int32_t dim,
    const sycl::range<3> offset,
    const sycl::nd_item<3> &item_ct1)
{
    size_t i[4];
    i[0] = item_ct1.get_global_id(2) + offset[2];
    i[1] = item_ct1.get_global_id(1) + offset[1];
    const size_t i2s= item_ct1.get_global_id(0) + offset[0];
    i[2] = i2s / ne[3];
    if (i[0] >= ne[0] || i[1] >= ne[1] || i2s >= ne[2]*ne[3] || i[2] >= ne[2]) {
        return;
    }
    //const int64_t i3 = i2s % ne3;
    i[3] = i2s % ne[3];

    //const T *x;
    ////if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
    //if(i[dim] < ne0[dim]){
    //    x = (const T *) (src0 + i[3]*nb0[3] +  i[2]*nb0[2] +  i[1]*nb0[1] +  i[0]*nb0[0]);
    //} else {
    //    //// これは微妙な前提だが…
    //    //x = (const T *) src1 + (i3-ne03)*nb13 +  (i2-ne02)*nb12 +  (i1-ne01)*nb11 +  (i0-ne00)*nb10;
    //    //x = (const T *) (src1 + (i3-(dim==3?ne0[3]:0))*nb1[3] +  (i2-(dim==2?ne0[2]:0))*nb1[2] +  (i1-(dim==1?ne0[1]:0))*nb1[1] +  (i0-(dim==0?ne0[0]:0))*nb1[0]);
    //    x = (const T *) (src1 + i[3]*nb1[3] + i[2]*nb1[2] + i[1]*nb1[1] + i[0]*nb1[0] - ne0[dim]*nb1[dim]);
    //}
    const T *x =  (const T *)(i[dim] < ne0[dim]
            ?src0 + i[3]*nb0[3] + i[2]*nb0[2] + i[1]*nb0[1] + i[0]*nb0[0]
            :src1 + i[3]*nb1[3] + i[2]*nb1[2] + i[1]*nb1[1] + i[0]*nb1[0] - ne0[dim]*nb1[dim]);
    //T *y = (T *)(dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);
    T *y = (T *)(dst + i[3] * nb[3] + i[2] * nb[2] + i[1] * nb[1] + i[0] * nb[0]);
     *y = *x;
    //(T *)(dst + i[3] * nb[3] + i[2] * nb[2] + i[1] * nb[1] + i[0] * nb[0]) = *x;
}
*/

template<typename T>
static void concat_T_non_cont_src0_offset(
    const char *src0,
    //const char *src1,
    char *dst,
    const size_t *ne0,
    const size_t *nb0,
    //const size_t *nb1,
    const size_t *ne,
    const size_t *nb,
    //const int32_t dim,
    const sycl::range<3> offset,
    const sycl::nd_item<3> &item_ct1)
{
    size_t i[4];
    i[0] = item_ct1.get_global_id(2) + offset[2];
    i[1] = item_ct1.get_global_id(1) + offset[1];
    const size_t i2s= item_ct1.get_global_id(0) + offset[0];
    i[2] = i2s / ne[3];
    if (i[0] >= ne0[0] || i[1] >= ne0[1] || i2s >= ne0[2]*ne[3] || i[2] >= ne0[2]) {
        return;
    }
    i[3] = i2s % ne[3];

    const T *x =  (const T *)(src0 + i[3]*nb0[3] + i[2]*nb0[2] + i[1]*nb0[1] + i[0]*nb0[0]);
    T *y = (T *)(dst + i[3] * nb[3] + i[2] * nb[2] + i[1] * nb[1] + i[0] * nb[0]);
    *y = *x;
}

template<typename T>
static void concat_T_non_cont_src1_offset(
    //const char *src0,
    const char *src1,
    char *dst,
    const size_t *ne0,
    //const size_t *nb0,
    const size_t *nb1,
    const size_t *ne,
    const size_t *nb,
    const int32_t dim,
    const sycl::range<3> offset,
    const sycl::nd_item<3> &item_ct1)
{
    size_t i[4];
    i[0] = item_ct1.get_global_id(2) + offset[2];
    i[1] = item_ct1.get_global_id(1) + offset[1];
    const size_t i2s= item_ct1.get_global_id(0) + offset[0];
    i[2] = i2s / ne[3];
    if (i[0] >= ne[0] || i[1] >= ne[1] || i2s >= ne[2]*ne[3] || i[2] >= ne[2]) {
        return;
    }
    i[3] = i2s % ne[3];

    const T *x =  (const T *)(src1 + i[3]*nb1[3] + i[2]*nb1[2] + i[1]*nb1[1] + i[0]*nb1[0] - ne0[dim]*nb1[dim]);
    T *y = (T *)(dst + i[3] * nb[3] + i[2] * nb[2] + i[1] * nb[1] + i[0] * nb[0]);
    *y = *x;
}


/*
// このロジックは元となるロジック
template<typename T>
static void concat_T_sycl_non_cont(
    //const char *src0, const char *src1, char *dst,
    //int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    //size_t  nb00, size_t  nb01, size_t  nb02, size_t  nb03,
    ////int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
    //size_t  nb10, size_t  nb11, size_t  nb12, size_t  nb13,
    //int64_t ne0,  int64_t ne1,  int64_t ne2,  int64_t ne3,
    //size_t  nb0,  size_t  nb1,  size_t  nb2,  size_t  nb3,

    const char *src0, const char *src1, char *dst,
    size_t ne00, size_t ne01, size_t ne02, size_t ne03,
    size_t  nb00, size_t  nb01, size_t  nb02, size_t  nb03,
    //int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
    size_t  nb10, size_t  nb11, size_t  nb12, size_t  nb13,
    size_t ne0,  size_t ne1,  size_t ne2,  size_t ne3,
    size_t  nb0,  size_t  nb1,  size_t  nb2,  size_t  nb3,
    int32_t dim,
    queue_ptr stream)
{
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    //GGML_SYCL_DEBUG("[SYCL] %s dim:%d", __func__, dim);
    //GGML_SYCL_DEBUG(" ne0(%ld, %ld, %ld, %ld)", ne00, ne01, ne02 ,ne03);
    //GGML_SYCL_DEBUG(" nb0(%zu, %zu, %zu, %zu)", nb00, nb01, nb02 ,nb03);
    //GGML_SYCL_DEBUG(" nb1(%zu, %zu, %zu, %zu)", nb10, nb11, nb12 ,nb13);
    //GGML_SYCL_DEBUG(" ne(%ld, %ld, %ld, %ld)", ne0, ne1, ne2 ,ne3);
    //GGML_SYCL_DEBUG(" nb(%zu, %zu, %zu, %zu)\n", nb0, nb1, nb2 ,nb3);

//    sycl::range<3> gridDim(ne3, ne2, ne1);
//    stream->parallel_for(sycl::nd_range<3>(gridDim, sycl::range<3>(1, 1, 1)),
//        [=](sycl::nd_item<3> item_ct1)
//        {
//            concat_T_non_cont<T>(
//                src0, src1, dst,
//                ne00, ne01, ne02, ne03,
//                nb00, nb01, nb02, nb03,
//                nb10, nb11, nb12, nb13,
//                ne0,  //ne1,  ne2,  ne3,
//                nb0,  nb1,  nb2,  nb3,
//                dim,
//                item_ct1);
//        }
//    );

            const size_t ane0[4] = {ne00, ne01, ne02, ne03};
            const size_t anb0[4] = {nb00, nb01, nb02, nb03};
            //size_t ne1[] = {ne10, ne11, ne12, ne13};
            const size_t anb1[4] = {nb10, nb11, nb12, nb13};
            const size_t ane[4] = {ne0, ne1, ne2, ne3};
            const size_t anb[4] = {nb0, nb1, nb2, nb3};

    sycl::range<3> world(ne3*ne2, ne1, ne0);
    sycl::range<3> local(1, 1, SYCL_CONCAT_WORK_GROUP_SIZE);
    ggml_sycl_looper(world, local, SYCL_CONCAT_WORK_GROUP_NUM, stream,
        [=](sycl::range<3> global, sycl::range<3> offset){

            //GGML_SYCL_DEBUG("[SYCL] %s global(%zu, %zu, %zu) offset(%zu, %zu, %zu)\n", __func__, global[0], global[1], global[2], offset[0], offset[1], offset[2]);


            stream->parallel_for(
                sycl::nd_range<3>(global, local),
                //sycl::nd_range<3>(gridDim, sycl::range<3>(1, 1, 1)),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(SYCL_CONCAT_SUB_GROUP_SIZE)]]
                {
                    concat_T_non_cont_offset<T>(
                        src0, src1, dst,
                        //ne00, ne01, ne02, ne03,
                        //nb00, nb01, nb02, nb03,
                        //nb10, nb11, nb12, nb13,
                        //ne0,  ne1,  ne2,  ne3,
                        //nb0,  nb1,  nb2,  nb3,
                        ane0, anb0, anb1, ane, anb,
                        dim,
                        offset,
                        item_ct1);
                }
            );

        }
    );

}
*/


// SRC0とSRC1のコピーを分割処理し、submit内の処理分岐を無くす。
// 残りの手段としては、最初は両方出力できるようにループさせ、SRC0の出力が終わったら
// SRC1だけ出力するロジックに切り替えるハイブリッド版を作成して様子を見ても悪くはないと思う。
template<typename T>
static void concat_T_sycl_non_cont_offset(
    const char *src0, const char *src1, char *dst,
    size_t ne00, size_t ne01, size_t ne02, size_t ne03,
    size_t nb00, size_t nb01, size_t nb02, size_t nb03,
    //size_t ne10, size_t ne11, size_t ne12, size_t ne13,
    size_t nb10, size_t nb11, size_t nb12, size_t nb13,
    size_t ne0,  size_t ne1,  size_t ne2,  size_t ne3,
    size_t nb0,  size_t nb1,  size_t nb2,  size_t nb3,
    int32_t dim,
    queue_ptr stream)
{
    //GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    GGML_SYCL_DEBUG("[SYCL] %s dim:%d", __func__, dim);
    GGML_SYCL_DEBUG(" ne(%zu, %zu, %zu, %zu)", ne0, ne1, ne2 ,ne3);
    GGML_SYCL_DEBUG(" ne0(%zu, %zu, %zu, %zu)", ne00, ne01, ne02 ,ne03);
    //GGML_SYCL_DEBUG(" ne1(%zu, %zu, %zu, %zu)", ne10, ne11, ne12 ,ne13);
    GGML_SYCL_DEBUG(" nb(%zu, %zu, %zu, %zu)", nb0, nb1, nb2 ,nb3);
    GGML_SYCL_DEBUG(" nb0(%zu, %zu, %zu, %zu)", nb00, nb01, nb02 ,nb03);
    GGML_SYCL_DEBUG(" nb1(%zu, %zu, %zu, %zu)\n", nb10, nb11, nb12 ,nb13);
    const size_t ane0[4] = {ne00, ne01, ne02, ne03};
    const size_t anb0[4] = {nb00, nb01, nb02, nb03};
    //size_t ne1[] = {ne10, ne11, ne12, ne13};
    const size_t anb1[4] = {nb10, nb11, nb12, nb13};
    const size_t ane[4] = {ne0, ne1, ne2, ne3};
    const size_t anb[4] = {nb0, nb1, nb2, nb3};

    // SRC0の分
    sycl::range<3> src0world(ne03*ne02, ne01, ne00);
    sycl::range<3> local(1, 1, SYCL_CONCAT_WORK_GROUP_SIZE);
    ggml_sycl_looper(src0world, local, SYCL_CONCAT_WORK_GROUP_NUM, stream,
        [=](sycl::range<3> global, sycl::range<3> offset){
            GGML_SYCL_DEBUG("[SYCL] %s SRC0 global(%zu, %zu, %zu) offset(%zu, %zu, %zu)\n", __func__, global[0], global[1], global[2], offset[0], offset[1], offset[2]);
            auto e =
            stream->parallel_for(
                sycl::nd_range<3>(global, local),
                //sycl::nd_range<3>(gridDim, sycl::range<3>(1, 1, 1)),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(SYCL_CONCAT_SUB_GROUP_SIZE)]]
                {
                    concat_T_non_cont_src0_offset<T>(
                        src0,
                        //src1,
                        dst,
                        //ane0, anb0, anb1, ane, anb,
                        ane0, anb0, ane, anb,
                        //dim,
                        offset,
                        item_ct1);
                }
            );
            SyclQueueEventWatcher::getInstance().SetEvent(e);

        }
    );
    // SRC1の分
    // dst基準で算出しているが、src1のne1xと同じになるはず…
    sycl::range<3> src1world((ne3 - (dim==3?ne03:0))*(ne2 - (dim==2?ne02:0)), ne1 - (dim==1?ne01:0), ne0 - (dim==0?ne00:0));
    //sycl::range<3> local(1, 1, SYCL_CONCAT_WORK_GROUP_SIZE);
    ggml_sycl_looper(src1world, local, SYCL_CONCAT_WORK_GROUP_NUM, stream,
        [=](sycl::range<3> global, sycl::range<3> offset){

            offset[2] = offset[2] + (dim==0?ne00:0);
            offset[1] = offset[1] + (dim==1?ne01:0);
            offset[0] = offset[0] + (dim==2?ne02:0) + (dim==3?ne03:0)*ne02;
            GGML_SYCL_DEBUG("[SYCL] %s SRC1 global(%zu, %zu, %zu) adjusted_offset(%zu, %zu, %zu)\n", __func__, global[0], global[1], global[2], offset[0], offset[1], offset[2]);
            auto e =
            stream->parallel_for(
                sycl::nd_range<3>(global, local),
                //sycl::nd_range<3>(gridDim, sycl::range<3>(1, 1, 1)),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(SYCL_CONCAT_SUB_GROUP_SIZE)]]
                {
                    concat_T_non_cont_src1_offset<T>(
                        //src0,
                        src1,
                        dst,
                        ane0,
                        //anb0,
                        anb1,
                        ane, anb,
                        dim,
                        offset,
                        item_ct1);
                }
            );
            SyclQueueEventWatcher::getInstance().SetEvent(e);

        }
    );

}


template <typename T>
void concat_impl_sycl(ggml_backend_sycl_context & ctx, ggml_tensor *dst)
{
    // この処理ggmlのソースの配列が入れ替わってたら多分アウトのような気がする。
    // ちゃんと条件見てるのかな？
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor *  src0   = dst->src[0];
    const ggml_tensor *  src1   = dst->src[1];
    queue_ptr            stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];
    GGML_SYCL_DEBUG("[SYCL] %s dim:%d\n", __func__, dim);

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const T * src0_d = (const T *) src0->data;
        const T * src1_d = (const T *) src1->data;
        T * dst_d = (T *) dst->data;
        size_t type_size = elem_size(dst->type);
        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<T>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    src0->ne[0], src0->ne[1], src0->ne[2],
                    dst->ne[0],  dst->ne[1],  dst->ne[2],
                    dim, stream);
            }

        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            //SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0).wait()));
            //SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d + size0 / type_size, src1_d, size1).wait()));

            SyclQueueEventWatcher::getInstance().WaitForSubmit();
            auto e = stream->memcpy(dst_d, src0_d, size0);
            SyclQueueEventWatcher::getInstance().SetEvent(e);

            SyclQueueEventWatcher::getInstance().WaitForSubmit();
            e = stream->memcpy(dst_d + size0 / type_size, src1_d, size1);
            SyclQueueEventWatcher::getInstance().SetEvent(e);
        }
    } else {
        //concat_T_sycl_non_cont<T>(
        concat_T_sycl_non_cont_offset<T>(
            (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
            src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            //src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
            dim,
            stream
        );
    }
}

void ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    switch (dst->type) {
    case GGML_TYPE_F32:
        concat_impl_sycl<float>(ctx, dst);
        break;
    case GGML_TYPE_F16:
        concat_impl_sycl<sycl::half>(ctx, dst);
        break;
#ifdef GGML_SYCL_HAS_BF16
    case GGML_TYPE_BF16:
        concat_impl_sycl<sycl::ext::oneapi::bfloat16>(ctx, dst);
        break;
#endif
    case GGML_TYPE_I32:
        concat_impl_sycl<int32_t>(ctx, dst);
        break;
    case GGML_TYPE_I16:
        concat_impl_sycl<int16_t>(ctx, dst);
        break;
    case GGML_TYPE_I64:
        concat_impl_sycl<int64_t>(ctx, dst);
        break;
    case GGML_TYPE_I8:
        concat_impl_sycl<int8_t>(ctx, dst);
        break;
    default:
        fprintf(stderr, "%s: unsupported types: dst: %s\n", __func__, ggml_type_name(dst->type));
        GGML_ASSERT(false);
    break;
    }
}
