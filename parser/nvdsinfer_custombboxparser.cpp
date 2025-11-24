/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <map>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))
#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

/* C-linkage to prevent name-mangling */         
extern "C"
bool NvDsInferParseCustomDatature (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);



extern "C"
bool NvDsInferParseCustomDatature (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    if(outputLayersInfo.size() != 1)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 1 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    // Host memory for "nms" which has 2 output bindings:
    // the order is bboxes and keep_count
    float* output = (float *) outputLayersInfo[0].buffer;
    const unsigned int outputSize = outputLayersInfo[0].inferDims.d[0];
    
    const float threshold = detectionParams.perClassThreshold[0];

    float* det;

    for (unsigned int i = 0; i < outputSize; i++) {
        det = output + i * 7;

        // Output format for each detection is stored in the below order
        // [ymin, xmin, ymax, xmax, confidence, class, valid]
        if ( det[4] < threshold || det[6] == 0) continue;
        assert((unsigned int) det[5] < detectionParams.numClassesConfigured);

        NvDsInferObjectDetectionInfo object;
            object.classId = (int) det[5];
            object.detectionConfidence = det[4];

            /* Clip object box co-ordinates to network resolution */
            object.left = CLIP(det[1] * networkInfo.width, 0, networkInfo.width - 1);
            object.top = CLIP(det[0] * networkInfo.height, 0, networkInfo.height - 1);
            object.width = CLIP((det[3] - det[1]) * networkInfo.width, 0, networkInfo.width - 1);
            object.height = CLIP((det[2] - det[0]) * networkInfo.height, 0, networkInfo.height - 1);

            objectList.push_back(object);
    }

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomDatature);


