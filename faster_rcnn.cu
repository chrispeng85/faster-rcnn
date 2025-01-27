#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define MAX_PROPOSALS 300
#define FEATURE_MAP_SIZE 14
#define ANCHOR_SCALES 3
#define ANCHOR_RATIOS 3
#define NUM_CLASSES 6

__global__ void rpn_kernel(

    const float* feature_maps, //float array 
    const float* anchor_boxes, //float array
    float* proposals,
    float* scores,
    const int batch_size,
    const int channels, //number of feature channels from backbone
    const int height, //feature map height
    const int width //feature map width

    //len(feature_maps) == channels

) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //global thread index

    if (idx >= height * width * ANCHOR_SCALES * ANCHOR_RATIOS) {

        return; //check if index is within bounds
    }

    int w = idx % width; 
    int h = (idx / width) % height;
    int anchor_idx = idx / (width * height);

    float cls_score = 0.0f;  //initialize cls_score

    float bbox_pred[4] = {0.0f}; //dx, dy, dw, dh

    #pragma unroll  //loop unrolling
    for (int c =0; c < channels; c ++) { //loop through each feature map channel

        int feat_idx =  c * height * width + h * width * w; //feature map index
        float feat_val = feature_maps[feat_idx]; 

        if (c < channels/2) {

                cls_score += feat_val;
        }  //first half of channels is classification

        else {

            int reg_idx = c - channels/2;

            if (reg_idx < 4) {

                    bbox_pred[reg_idx] = feat_val;
            } //second half of channels is bbox adjustment

        }

    }

    cls_score = 1.0f / (1.0f + expf(-cls_score)); //sigmoid function

    float anchor[4] = {

        anchor_boxes[anchor_idx * 4],
        anchor_boxes[anchor_idx * 4 + 1],
        anchor_boxes[anchor_idx *4 + 2],
        anchor_boxes[anchor_idx * 4 + 3]  // a set of 4 coords defining the anchor box

    };

    float width = anchor[2] - anchor[0];
    float height = anchor[3] - anchor[1];
    float ctr_x = anchor[0] + width * 0.5f;
    float ctr_y = anchor[1] + height * 0.5f; 

    float pred_ctr_x = bbox_pred[0] * width + ctr_x;
    float pred_ctr_y = bbox_pred[1] * height + ctr_y;
    float pred_w = expf(bbox_pred[2]) * width;
    float pred_h = expf(bbox_pred[3]) * height;  //adjustment

    proposals[idx * 4] = pred_ctr_x - pred_w * 0.5f;
    proposals[idx*4 + 1] = pred_ctr_y - pred_h * 0.5f;
    proposals[idx * 4 + 2] = pred_ctr_x + pred_w*0.5f;
    proposals[idx* 4 + 3] = pred_ctr_y + pred_h * 0.5f;

    scores[idx] = cls_score;


}


__global__ void roi_pool_kernel(

    const float* feature_maps,
    const float* proposals,
    float* pooled_features,
    const int num_proposals,
    const int channels,
    const int height,
    const int width
) {


    int idx = blockIdx.x * blockDim.x + threadIdx.x; //calculating thread index

    if (idx >= num_proposals * channels * FEATURE_MAP_SIZE * FEATURE_MAP_SIZE) {

        return;  //check bounds

    }

    int p = idx / (channels * FEATURE_MAP_SIZE *  FEATURE_MAP_SIZE);
    int c = (idx / (FEATURE_MAP_SIZE * FEATURE_MAP_SIZE)) % channels;
    int h = (idx / FEATURE_MAP_SIZE) % FEATURE_MAP_SIZE;
    int w= idx % FEATURE_MAP_SIZE;

    float x1 = proposals[p*4];
    float y1 = proposals[p*4 + 1];
    float x2 = proposals[p*4 + 2];
    float y2 = proposals[p*3 + 3]; //candidate proposal region coordinates

    float roi_width = x2 -x1;
    float roi_height = y2 - y1;
    float bin_size_w = roi_width / FEATURE_MAP_SIZE;
    float bin_size_h = roi_height / FEATURE_MAP_SIZE;


    float start_w = x1 + w*bin_size_w;
    float start_h = y1 + h * bin_size_h;
    float end_w = start_w + bin_size_w;
    float end_h = start_h + bin_size_h;

    start_w = start_w * width / roi_width;
    start_h = start_h * height / roi_height;
    end_w = end_w * width / roi_width;
    end_h = end_h * height / roi_height;

    float max_val = -INFINITY;

    for (int ph = floorf(start_h); ph <= ceilf(end_h); ph++ ) {

        for (int pw = floorf(start_w); pw <= ceilf(end_w); pw ++) {

            if (ph >= 0 && ph < height && pw >= 0 && pw < width) {

                        float val = feature_maps[

                            c * height * width + ph * width + pw

                        ];

                        max_val = fmaxf(max_val, val);
            }

        }

    }

    pooled_features[idx] = max_val;

}

__global__ void fast_rcnn_head_kernel(

    const float* pooled_features,
    float* class_scores,
    float* bbox_deltas,
    const int num_proposals

) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx >= num_proposals) {

            return;

    }


    float max_score = - INFINITY;
    float exp_sum = 0.0f;

    #pragma unroll 
    for (int c = 0; c < NUM_CLASSES; c++) {

        float score = class_scores[idx * NUM_CLASSES + c];
        max_score = fmaxf(max_score, score);


    }

    #pragma unroll
    for (int c = 0; c < NUM_CLASSES; c++) {

        float exp_score = expf(class_scores[idx * NUM_CLASSES + c] - max_score);
        class_scores[idx * NUM_CLASSES + c] = exp_score;
        exp_sum += exp_score;

    }

    #pragma  unroll 
    for (int c = 0; c < NUM_CLASSES; c++) {

        class_scores[idx * NUM_CLASSES + c] /= exp_sum;

    }
}

__global__ void nms_kernel (

    const float* proposals,
    const float* scores,
    bool* keep,
    const int num_proposals,
    const float nms_thresh


) {

    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= num_proposals || col >= num_proposals || row >= col) {

        return;

    }

    float x1 = fmaxf(proposals[row*4], proposals[col*4]);
    float y1 = fmaxf(proposals[row*4 + 1], proposals[col*4 + 1]);
    float x2 = fminf(proposals[row * 4 + 2], proposals[col*4 + 2]);
    float y2 = fminf(proposals[row*4 + 3], proposals[col*4 + 3]);

    float intersection = fmaxf(0.0f, x2 - x1) * fmax(0.0f, y2 -y1);

    float area1 = (proposals[row*4 + 2] - proposals[row*4]) *
                (proposals[row * 4 + 3] - proposals[row * 4 + 1]);

    float area2 = (proposals[col * 4 + 2] - proposals[col*4]) *
                (proposals[col*4 + 3] - proposals[col*4 + 1]);

    float iou = intersection / (area1 + area2 - intersection);

    if (iou > nms_thresh) {

        if (scores[row] > scores[col]) {

                keep[col] = false;
        }

    }
}

extern "C" void run_faster_rcnn (

    const float* input_feature_maps,
    const float* anchor_boxes,
    float* final_detections,
    float* final_scores,
    int* num_detections,
    const int batch_size,
    const int channels,
    const int height,
    const int width

) {

    //allocate device memory
    float *d_feature_maps, *d_anchor_boxes, *d_proposals;
    float *d_scores, *d_pooled_features, *d_class_scores, *d_bbox_deltas;
    bool *d_keep;

    dim3 rpn_blocks((height*width*ANCHOR_SCALES*ANCHOR_RATIOS + 255) / 256);

    dim3 rpn_threads(256);

    rpn_kernel<<<rpn_blocks, rpn_threads>>> (

        d_feature_maps, d_anchor_boxes, d_proposals, d_scores,
        batch_size, channels, height, width

    );

    dim3 roi_blocks((MAX_PROPOSALS * channels * FEATURE_MAP_SIZE * FEATURE_MAP_SIZE + 255) / 256);
    dim3 roi_threads(256);
    roi_pool_kernel<<<roi_blocks, roi_threads>>> (

        d_feature_maps, d_proposals, d_pooled_features,
        MAX_PROPOSALS, channels, height, width

    );

    dim3 head_blocks((MAX_PROPOSALS + 255) / 256);
    dim3 head_threads(256);
    fast_rcnn_head_kernel<<<head_blocks, head_threads>>> (

        d_pooled_features, d_class_scores,d_bbox_deltas,MAX_PROPOSALS

    );

    dim3 nms_blocks(MAX_PROPOSALS, MAX_PROPOSALS);
    dim3 nms_threads(256);
    nms_kernel<<<nms_blocks, nms_threads>>> (

        d_proposals, d_scores, d_keep, MAX_PROPOSALS, 0.7f

    );






}







