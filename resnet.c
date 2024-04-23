/*
gcc -o resnet -g  resnet.c -lm -fopenmp
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {

} Context;

typedef struct {
    float* x;
    float* xb;
    float* xb1;
    float* xb2;
    float* xb3;
    int* xb4;

    float *hidden0_0;
    float *hidden0_1;
    float *hidden1_0;
    float *hidden1_1;
    float *hidden1_2;
    float *hidden2_0;
    float *hidden2_1;
    float *hidden3_0;
    float *hidden3_1;

    int N;
    int H_in;
    int W_in;
    int H_out;
    int W_out;

    int max_mem;
} RunState;


typedef struct {
    float* embedder; // (64, 3, 7, 7)
    float *conv_weight;
    float *norm_weight;
    float *norm_bias;
    float *norm_running_mean;
    float *norm_running_var;
} ResNetWeights;

typedef struct {
    int embedding_size;
    int depths[4];
    int hidden_sizes[4];
    char *layer_type;
    int num_channels;
    char *activation;
    char *hidden_act;
    int downsample_in_bottleneck;
    int num_labels;
} ResNetConfig;

typedef struct {
    ResNetConfig config;
    ResNetWeights weights;
    float* params_memory;
    int num_parameters;

    int batch_size;
    float* inputs;
    RunState state;
    
} ResNet;

void malloc_run_state(RunState* s, ResNetConfig* p) {
    // // we calloc instead of malloc to keep valgrind happy
    // int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    // s->x = calloc(p->dim, sizeof(float));
    // s->xb = calloc(p->dim, sizeof(float));
    // s->xb2 = calloc(p->dim, sizeof(float));
    // s->hb = calloc(p->hidden_dim, sizeof(float));
    // s->hb2 = calloc(p->hidden_dim, sizeof(float));
    // s->q = calloc(p->dim, sizeof(float));
    // s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    // s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    // s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    // s->logits = calloc(p->vocab_size, sizeof(float));
    // // ensure all mallocs went fine
    // if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
    //  || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
    //     fprintf(stderr, "malloc failed!\n");
    //     exit(EXIT_FAILURE);
    // }
}

void free_run_state(RunState* s) {
    // free(s->x);
    // free(s->xb);
    // free(s->xb2);
    // free(s->hb);
    // free(s->hb2);
    // free(s->q);
    // free(s->att);
    // free(s->logits);
    // free(s->key_cache);
    // free(s->value_cache);
}

void resnet_build_from_checkpoint(ResNet *model, char* checkpoint_path) {
    FILE *model_file = fopen(checkpoint_path, "rb");
    if (model_file == NULL) {
        printf("Error opening model file\n");
    }

    size_t file_size = 0;
    fseek(model_file, 0, SEEK_END);
    file_size = ftell(model_file);
    fseek(model_file, 0, SEEK_SET);
    printf("file_size is %ld\n", file_size);

    int model_magic;
    fread(&model_magic, sizeof(int), 1, model_file);
    if (model_magic != 20240416) {
        printf("Bad magic model file");
    }
    printf("model magic is %d\n", model_magic);

    fread(model->config.depths, sizeof(int), 4, model_file);
    if (model->config.depths == NULL) {
        printf("Bad depths in model file");
    }
    printf("depths: %d %d %d %d\n", model->config.depths[0], model->config.depths[1], 
                                    model->config.depths[2], model->config.depths[3]);

    fread(&model->config.embedding_size, sizeof(int), 1, model_file);
    printf("embedding_size: %d\n", model->config.embedding_size);
    
    
    fread(model->config.hidden_sizes, sizeof(int), 4, model_file);
    printf("hidden_sizes: %d %d %d %d\n", model->config.hidden_sizes[0], model->config.hidden_sizes[1], 
                                    model->config.hidden_sizes[2], model->config.hidden_sizes[3]);
    
    int head_size = 10;
    size_t model_size = file_size - head_size * sizeof(int);

    model->num_parameters = model_size / sizeof(float);
    printf("num_parameters: %d\n", model->num_parameters);

    model->params_memory = (float*)malloc(model_size);
    fread(model->params_memory, sizeof(float), model->num_parameters, model_file);
    // for (int i = 0; i < 64; i++) {
    //     printf("weight: %f ", *(model->params_memory+i));
    // }
    model->weights.embedder = model->params_memory;
}


typedef struct {
    // bchw
    int batch;
    int channel;
    int height;
    int width;
    float* data;
} Image;

void read_image(Image *img, char* img_path) {
    FILE *img_file = fopen(img_path, "rb");
    if (img_file == NULL) {
        printf("Error opening image file\n");
    }

    int headers[4];
    fread(headers, sizeof(int), 4, img_file);
    img->batch = headers[0];
    img->channel = headers[1];
    img->height = headers[2];
    img->width = headers[3];
    
    printf("image shape: %d %d %d %d\n", img->batch, img->channel, img->height, img->width);

    img->data = (float*)malloc(img->batch * img->channel * img->width * img->height * sizeof(float));
    fread(img->data, sizeof(float), img->batch * img->channel * img->width * img->height, img_file);
    // for (int i = img->batch * img->channel * img->height * img->width - 320; i < img->batch * img->channel * img->height * img->width; i++) {
    //     printf("%f ", *(img->data + i));
    // }
}

// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
void conv2d_forwardV2(float* output, float* input, float *kernel_weight, float* bias, 
                      int N, int C_in, int H_in, int W_in, int C_out, int H_out, int W_out, 
                      int kernel_size, int stride, int padding, int dilation, int is_bias, char* padding_mode) {
    if (stride < 0) {
        stride = 1;
    }
    // printf("conv2d_forwardV2 N:%d C_in:%d H_in:%d W_in:%d C_out:%d H_out:%d W_out:%d kernel_size:%d stride:%d padding:%d\n", 
    //         N, C_in, H_in, W_in, C_out, H_out, W_out, 
    //         kernel_size,  stride, padding);
    for (int n = 0; n < N; n++) {
        int c_out = 0;
        #pragma omp parallel for private(c_out)
        for (c_out = 0; c_out < C_out; c_out++) {
            for (int h_in = 0; h_in < H_in + 2 * padding - kernel_size + 1; h_in += stride) {
                for (int w_in = 0; w_in < W_in + 2 * padding - kernel_size + 1; w_in += stride) {
                    int offset_out = n * C_out * H_out * W_out
                                   + c_out * H_out * W_out 
                                   + h_in / stride * W_out
                                   + w_in / stride;
                    float value = 0.0f;
                    for (int c_in = 0; c_in < C_in; c_in++) {
                        for (int k_i = 0; k_i < kernel_size; k_i++) {
                            for (int k_j = 0; k_j < kernel_size; k_j++){
                                int offset_kernel = c_out * C_in * kernel_size * kernel_size
                                                  + c_in * kernel_size * kernel_size
                                                  + k_i * kernel_size + k_j;
                                float input_v = 0.0f;
                                if (h_in + k_i >= padding && h_in + k_i < H_in + padding && w_in + k_j >= padding && w_in + k_j < W_in + padding) {
                                    int offset_in = n * C_in * H_in * W_in
                                                  + c_in * H_in * W_in
                                                  + (h_in - padding) * W_in
                                                  + (w_in - padding)
                                                  + k_i * W_in + k_j;
                                    input_v = input[offset_in];
                                
                                }
                                value += input_v * (*(kernel_weight + offset_kernel));
                            }
                        }                   
                    }
                    output[offset_out] = value;
                    // if (offset_out < N * C_out * H_out * W_out && offset_out >= N * C_out * H_out * W_out - 640) {
                    // if (offset_out < 640) {
                    //     printf("conv2d_forwardV2 n:%d c_out:%d h_out:%d w_out:%d output[%d]: %f\n", n, c_out, h_in/stride, w_in/stride, offset_out, value);
                    // }
                }
            }
        }
    }
}

// https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
void relu(float* output, float* input, int N, int C, int H, int W) {
    // printf("relu N:%d C:%d H:%d W:%d\n", N, C, H, W);
    for (int n = 0; n < N; n++) {
        int c = 0;
        #pragma omp parallel for private(c)
        for (c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int offset = n * C * H * W
                               + c * H * W
                               + h * W
                               + w;
                    output[offset] = 0.0f;
                    if (input[offset] > 0) {
                        output[offset] = input[offset];
                    }
                }
            }
        }
    }
}

// https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
void batchnorm2d_forwardV2(float* output, float* input, float *weight, float *bias, float *running_mean, float *running_var,
                         int N, int C, int H, int W) {
    // printf("batchnorm2d_forwardV2 N:%d C:%d H:%d W:%d\n", N, C, H, W);
    float eps = 1e-05;

    for(int n = 0; n < N; n++) {
        int c = 0;
        #pragma omp parallel for private(c)
        for (c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int offset = n * C * H * W
                                 + c * H * W
                                 + h * W
                                 + w;
                    output[offset] = (input[offset] - (*(running_mean + c))) 
                                     / sqrtf(*(running_var + c) + eps) 
                                     * (*(weight + c)) 
                                     + (*(bias + c));
                    // if (offset < N * C * H * W && offset >= N * C * H * W - 640) {
                    // // if (offset < 640) {
                    //     printf("batchnorm2d_forwardV2 n:%d c:%d h:%d w:%d output[%d]: %f\n", n, c, h, w, offset, output[offset]);
                    // }
                }
            }
        }
    }
}

// https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
void identity(float* output, float* input, int N, int C, int H, int W) {
    // printf("identity N:%d C:%d H:%d W:%d\n", N, C, H, W);
    for (int n = 0; n < N; n++) {
        int c = 0;
        #pragma omp parallel for private(c)
        for (c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int offset = n * C * H * W
                               + c * H * W
                               + h * W
                               + w;
                    output[offset] = input[offset];
                }
            }
        }
    }
}

// https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
void maxpool2d_forward(float* output, float* input, int N, int C, int H, int W, int H_out, int W_out,
                       int kernel_size, int stride, int padding, int dilation, int ceil_mode) {
    // printf("maxpool2d_forward B:%d C:%d H:%d W: %d H_out:%d W_out:%d kernel_size:%d stride:%d padding:%d\n", N, C, H, W, H_out, W_out, kernel_size, stride, padding);
    for(int n = 0; n < N; n++) {
        int c = 0;
        #pragma omp parallel for private(c)
        for (c = 0; c < C; c++) {
            for (int h = 0; h < H + 2 * padding - kernel_size + 1; h += stride) {
                for (int w = 0; w < W + 2 * padding - kernel_size + 1; w += stride) {
                    
                    float value = -INFINITY;
                    for (int k_i = 0; k_i < kernel_size; k_i++) {
                        for (int k_j = 0; k_j < kernel_size; k_j++){
                            if (h + k_i >= padding && h + k_i < H + 2 * padding && w + k_j >= padding && w + k_j < W + 2 * padding) {
                                int offset_in = n * C * H * W
                                  + c * H * W
                                  + (h - padding) * W
                                  + (w - padding)
                                  + k_i * W + k_j;
                                if (value < input[offset_in]) {
                                    value = input[offset_in];
                                }
                            }
                            

                        }
                    }
                    int offset_out = n * C * H_out * W_out
                                   + c * H_out * W_out
                                   + (h / stride) * W_out
                                   + w / stride;
                    output[offset_out] = value;
                    // if (offset_out < N * C * H_out * W_out && offset_out >= N * C * H_out * W_out - 640) {
                    // // if (offset_out < 640) {
                    //     printf("b:%d c_out:%d h_out:%d w_out:%d output[%d]: %f\n", n, c, h/stride, w/stride, offset_out, output[offset_out]);
                    // }
                }
            }
        }
    }
}

// https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
void adaptiveavgpool2d_1x1_forward(float* output, float* input, int N, int C, int H, int W)  {
    // printf("adaptiveavgpool2d_1x1_forward B:%d C:%d H:%d W: %d \n", N, C, H, W);
    for(int n = 0; n < N; n++) {
        int c = 0;
        #pragma omp parallel for private(c)
        for (c = 0; c < C; c++) {
            float ss = 0.0f;
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int offset_in = n * C * H * W
                                   + c * H * W
                                   + h * W
                                   + w;
                    ss += input[offset_in];
                }
            }
            int offset_out = n * C + c;
            // printf("adaptiveavgpool2d_1x1_forward n:%d c:%d ss:%f %d\n", n, c, ss, offset_out);
            output[offset_out] = ss / (H * W);
                    // if (offset_out < N * C * H_out * W_out && offset_out >= N * C * H_out * W_out - 640) {
                    // // if (offset_out < 640) {
                    //     printf("b:%d c_out:%d h_out:%d w_out:%d output[%d]: %f\n", n, c, h/stride, w/stride, offset_out, output[offset_out]);
                    // }
        }
    }
}

// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
void linear_forward(float* output, float* input, float *weight, float *bias, int M, int N, int K)  {
    // printf("linear_forward M:%d N:%d K:%d\n", M, N, K);
    for (int i = 0; i < M; i++) {
        int j = 0;
        #pragma omp parallel for private(j)
        for (j = 0; j < N; j++) {
            int offset_out = i * N + j;
            int offset_bias = j;
            output[offset_out] = 0;
            for (int k = 0; k < K; k++) {
                int offset_in = i * K + k;
                int offset_weight = j * K + k;
                
                output[offset_out] += input[offset_in] * weight[offset_weight];
            }
            output[offset_out] += bias[offset_bias];
        }
    }
}

void softmax_forward(float* input, int M, int N) {
    // printf("softmax_forward M:%d N:%d\n", M, N);
    int m;
    #pragma omp parallel for private(m)
    for (m = 0; m < M; m++) {
        float max_val = input[m * N];
        for (int n = 1; n < N; n++) {
            if (input[m * N + n] > max_val) {
                max_val = input[m * N + n];
            }
        }

        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            input[m * N + n] = expf(input[m * N + n] - max_val);
            sum += input[m * N + n];
        }
        for (int n = 0; n < N; n++) {
            input[m * N + n] /= sum;
        }
    }
}

void argmax_forward(int* output, float* input, int M, int N) {
    // printf("argmax_forward M:%d N:%d\n", M, N);
    int m = 0;
    #pragma omp parallel for private(m)
    for (m = 0; m < M; m++) {
        int v = 0;
        for (int n = 1; n < N; n++) {
           if (input[m * N + n] > input[m * N + v]) {
               v = n;
           }
        }
        output[m] = v;
    }
}

void ResNetConvLayerV2(Context *ctx, float *output, float *input, RunState *s, ResNetWeights *w, int in_channels, int out_channels, int kernel_size, int stride, char *activation) {
    if (kernel_size < 0) {
        kernel_size = 3;
    }

    if (stride < 0) {
        stride = 1;
    }

    if (activation == NULL) {
        activation = "relu";
    }

    int padding = kernel_size / 2;
    s->H_out = (s->H_in + 2 * padding - kernel_size + 1) / stride;
    s->W_out = (s->W_in + 2 * padding - kernel_size + 1) / stride;
    w->norm_weight = w->conv_weight + in_channels * out_channels * kernel_size * kernel_size;
    w->norm_bias =  w->norm_weight + out_channels;
    w->norm_running_mean = w->norm_bias + out_channels;
    w->norm_running_var = w->norm_running_mean + out_channels;
    // if (s->N * out_channels * s->H_out * s->W_out >= s->max_mem) {
    //     s->max_mem = s->N * out_channels * s->H_out * s->W_out;
    //     printf("N:%d C:%d H:%d W:%d\n", s->N, out_channels, s->H_out, s->W_out);
    // }
    conv2d_forwardV2(s->hidden0_0, input, w->conv_weight, NULL, s->N, in_channels, s->H_in, s->W_in, out_channels, s->H_out, s->W_out, kernel_size, stride, padding, 0, 0, NULL);
    if (activation == "relu") {
        batchnorm2d_forwardV2(s->hidden0_1, s->hidden0_0, w->norm_weight, w->norm_bias, w->norm_running_mean, w->norm_running_var, s->N, out_channels, s->H_out, s->W_out);
        relu(output, s->hidden0_1, s->N, out_channels, s->H_out, s->W_out);
    } else {
        batchnorm2d_forwardV2(output, s->hidden0_0, w->norm_weight, w->norm_bias, w->norm_running_mean, w->norm_running_var, s->N, out_channels, s->H_out, s->W_out);
    }
    
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/resnet/modeling_resnet.py
void ResNetEmbeddingsV2(Context *ctx, float *output, float *input, RunState *s, ResNetWeights *w, ResNetConfig *p) {
    ResNetConvLayerV2(ctx, s->hidden1_0, input, s, w, p->num_channels, p->embedding_size, 7, 2, p->hidden_act);
    int kernel_size = 3;
    int stride = 2;
    int padding = 1;
    s->H_in = s->H_out;
    s->W_in = s->W_out;
    s->H_out = (s->H_in + 2 * padding - kernel_size + 1) / stride;
    s->W_out = (s->W_in + 2 * padding - kernel_size + 1) / stride;
    maxpool2d_forward(output, s->hidden1_0, s->N, p->embedding_size, s->H_in, s->W_in, s->H_out, s->W_out, kernel_size, stride, padding, 0, 0);
    // for (int i = 0; i < 320; i++) {
    //     printf("%d=%f ", i, *(output + i));
    // }
}

void ResNetShortCutV2(Context *ctx, float *output, float *input, RunState *s, ResNetWeights *w, int in_channels, int out_channels, int stride) {
    if (stride < 0) {
        stride = 2;
    }
    int kernel_size = 1;
    int padding = 0;
    s->H_out = (s->H_in + 2 * padding - kernel_size + 1) / stride;
    s->W_out = (s->W_in + 2 * padding - kernel_size + 1) / stride;
    w->norm_weight = w->conv_weight + in_channels * out_channels * kernel_size * kernel_size;
    w->norm_bias =  w->norm_weight + out_channels;
    w->norm_running_mean = w->norm_bias + out_channels;
    w->norm_running_var = w->norm_running_mean + out_channels;
    // if (s->N * out_channels * s->H_out * s->W_out >= s->max_mem) {
    //     s->max_mem = s->N * out_channels * s->H_out * s->W_out;
    //     printf("N:%d C:%d H:%d W:%d\n", s->N, out_channels, s->H_out, s->W_out);
    // }
    conv2d_forwardV2(s->hidden0_0, input, w->conv_weight, NULL, s->N, in_channels, s->H_in, s->W_in, out_channels, s->H_out, s->W_out, kernel_size, stride, padding, 0, 0, NULL);
    batchnorm2d_forwardV2(output, s->hidden0_0, w->norm_weight, w->norm_bias, w->norm_running_mean, w->norm_running_var, s->N, out_channels, s->H_out, s->W_out);
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/resnet/modeling_resnet.py
void ResNetBottleNeckLayerV2(Context *ctx, float *output, float *input, RunState *s, ResNetWeights *w,
                            int in_channels, int out_channels, int stride, char *activation, int reduction, int downsample_in_bottleneck) {

    int H_in = s->H_in;
    int W_in = s->W_in;

    int kernel_size = 0;
    int real_stride = stride;
    if (stride < 0) {
        real_stride = 1;
    }
    if (activation == NULL) {
        activation = "relu";
    }
    if (reduction < 0) {
        reduction = 4;
    }
    int padding = 0;

    int reduces_channels = out_channels / reduction;
    // printf("++++++++++++in_channels:%d out_channels:%d stride:%d\n", in_channels, out_channels, real_stride);
    if (in_channels != out_channels || real_stride != 1) {
        ResNetShortCutV2(ctx, s->hidden1_0, input, s, w, in_channels, out_channels, stride); 
        w->conv_weight = w->norm_running_var + out_channels;
    } else {
        identity(s->hidden1_0, input, s->N, in_channels, s->H_in, s->W_in);
    }
     
    // if (in_channels == 256 && out_channels == 512 && real_stride == 2) {
    // for (int i = s->N * out_channels * s->H_out * s->W_out - 320; i < s->N * out_channels * s->H_out * s->W_out; i++) {
    // for (int i = 0; i < 320; i++) {
    //     printf("%d=%f ", i, *(s->hidden1_0 + i));
    // }
    // printf("\n\n");
    // for (int i = 0; i < 512; i++) {
    //     printf("%d=%f ", i, *(w->norm_running_var + 256 + i));
    // }
    // }

    kernel_size = 1;
    if (downsample_in_bottleneck != 0) {
        real_stride = stride;
    } else {
        real_stride = 1;
    }
    padding = 0;
    s->H_in = H_in;
    s->W_in = W_in;
    // conv2d_forwardV2 N:2 C_in:64 H_in:120 W_in:160 C_out:64 H_out:120 W_out:160 kernel_size:1 stride:1 padding:0
    ResNetConvLayerV2(ctx, s->hidden1_1, input, s, w, in_channels, reduces_channels, kernel_size, real_stride, NULL);
    // for (int i = s->N * reduces_channels * s->H_out * s->W_out - 320; i < s->N * reduces_channels * s->H_out * s->W_out; i++) {
    // // for (int i = 0; i < 320; i++) {
    //     printf("%d=%f ", i, *(s->hidden1_1 + i));
    // }

    w->conv_weight = w->norm_running_var + reduces_channels;
    if (downsample_in_bottleneck != 0) {
        real_stride = 1;
    } else {
        real_stride = stride;
    }
    s->H_in = s->H_out;
    s->W_in = s->W_out;
    ResNetConvLayerV2(ctx, s->hidden1_2, s->hidden1_1, s, w, reduces_channels, reduces_channels, -1, real_stride, NULL);
    // // for (int i = s->N * reduces_channels * s->H_out * s->W_out - 320; i < s->N * reduces_channels * s->H_out * s->W_out; i++) {
    // for (int i = 0; i < 320; i++) {
    //     printf("%d=%f ", i, *(s->hidden1_2 + i));
    // }

    w->conv_weight = w->norm_running_var + reduces_channels;
    s->H_in = s->H_out;
    s->W_in = s->W_out;
    ResNetConvLayerV2(ctx, s->hidden1_1, s->hidden1_2, s, w, reduces_channels, out_channels, 1, -1, "None");
    // // for (int i = s->N * out_channels * s->H_out * s->W_out - 320; i < s->N * out_channels * s->H_out * s->W_out; i++) {
    // for (int i = 0; i < 320; i++) {
    //     printf("%d=%f ", i, *(s->hidden1_1 + i));
    // }

    // residual
    

    if (activation == "relu") {
        for (int i = 0; i < s->N * out_channels * s->H_out * s->W_out; i++) {
            *(s->hidden1_2 + i) = (*(s->hidden1_0 + i)) + (*(s->hidden1_1 + i));
        }
        relu(output, s->hidden1_2, s->N, out_channels, s->H_out, s->W_out);
    } else {
        for (int i = 0; i < s->N * out_channels * s->H_out * s->W_out; i++) {
            *(output + i) = (*(s->hidden1_0 + i)) + (*(s->hidden1_1 + i));
        }
    }
    // // for (int i = s->N * out_channels * s->H_out * s->W_out - 320; i < s->N * out_channels * s->H_out * s->W_out; i++) {
    // for (int i = 0; i < 320; i++) {
    //     printf("%d=%f ", i, *(output + i));
    // }
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/resnet/modeling_resnet.py
void ResNetStageV2(Context *ctx, float *output, float *input, RunState *s, ResNetWeights *w, ResNetConfig *p, int in_channels, 
                   int out_channels, int stride, int depth) {
    if (stride < 0) {
        stride = 2;
    }

    if (depth < 0) {
        depth = 2;
    }

    ResNetBottleNeckLayerV2(ctx, s->hidden2_0, input, s, w, in_channels, out_channels, stride, p->hidden_act, -1, p->downsample_in_bottleneck);
    // // for (int i = s->N * out_channels * s->H_out * s->W_out - 320; i < s->N * out_channels * s->H_out * s->W_out; i++) {
    // for (int i = 0; i < 320; i++) {
    //     // printf("%d=%f ", i, (*(s->residual + i)) + (*(s->hidden4 + i)));
    //     printf("%d=%f ", i, *(s->hidden2_0 + i));
    // }

    for (int i = 0; i < depth - 2; i++) {
        s->H_in = s->H_out;
        s->W_in = s->W_out;
        w->conv_weight = w->norm_running_var + out_channels;
        if (i % 2 == 0) {
            ResNetBottleNeckLayerV2(ctx, s->hidden2_1, s->hidden2_0, s, w, out_channels, out_channels, -1, p->hidden_act, -1, 0);
        } else {
            ResNetBottleNeckLayerV2(ctx, s->hidden2_0, s->hidden2_1, s, w, out_channels, out_channels, -1, p->hidden_act, -1, 0);
        }
    }
    
    // // for (int i = s->N * out_channels * s->H_out * s->W_out - 320; i < s->N * out_channels * s->H_out * s->W_out; i++) {
    // for (int i = 0; i < 320; i++) {
    //     // printf("%d=%f ", i, (*(s->residual + i)) + (*(s->hidden4 + i)));
    //     printf("%d=%f ", i, *(s->hidden2_0 + i));
    // }

    s->H_in = s->H_out;
    s->W_in = s->W_out;
    w->conv_weight = w->norm_running_var + out_channels;
    if ((depth-2) % 2 == 0) {
        ResNetBottleNeckLayerV2(ctx, output, s->hidden2_0, s, w, out_channels, out_channels, -1, p->hidden_act, -1, 0);
    } else {
        ResNetBottleNeckLayerV2(ctx, output, s->hidden2_1, s, w, out_channels, out_channels, -1, p->hidden_act, -1, 0);
    }
    // // for (int i = s->N * out_channels * s->H_out * s->W_out - 320; i < s->N * out_channels * s->H_out * s->W_out; i++) {
    // for (int i = 0; i < 320; i++) {
    //     printf("%d=%f ", i, *(output + i));
    // }
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/resnet/modeling_resnet.py
void ResNetEncoderV2(Context *ctx, float *output, float *input, RunState *s, ResNetWeights *w, ResNetConfig *p) {
    ResNetStageV2(ctx, s->hidden3_0, input, s, w, p, p->embedding_size, p->hidden_sizes[0], 1, p->depths[0]);
    // // for (int i = s->N * p->hidden_sizes[0] * s->H_out * s->W_out - 320; i < s->N * p->hidden_sizes[0] * s->H_out * s->W_out; i++) {
    // for (int i = 0; i < 320; i++) {
    //     printf("%d=%f ", i, *(s->hidden3_0 + i));
    // }

    s->H_in = s->H_out;
    s->W_in = s->W_out;
    w->conv_weight = w->norm_running_var + p->hidden_sizes[0];
    ResNetStageV2(ctx, s->hidden3_1, s->hidden3_0, s, w, p, p->hidden_sizes[0], p->hidden_sizes[1], -1, p->depths[1]);
    // for (int i = s->N * p->hidden_sizes[1] * s->H_out * s->W_out - 320; i < s->N * p->hidden_sizes[1] * s->H_out * s->W_out; i++) {
    // // for (int i = 0; i < 320; i++) {
    //     printf("%d=%f ", i, *(s->hidden3_1 + i));
    // }

    s->H_in = s->H_out;
    s->W_in = s->W_out;
    w->conv_weight = w->norm_running_var + p->hidden_sizes[1];
    ResNetStageV2(ctx, s->hidden3_0, s->hidden3_1, s, w, p, p->hidden_sizes[1], p->hidden_sizes[2], -1, p->depths[2]);
    // // for (int i = s->N * p->hidden_sizes[2] * s->H_out * s->W_out - 320; i < s->N * p->hidden_sizes[2] * s->H_out * s->W_out; i++) {
    // for (int i = 0; i < 320; i++) {
    //     printf("%d=%f ", i, *(s->hidden3_0 + i));
    // }

    s->H_in = s->H_out;
    s->W_in = s->W_out;
    w->conv_weight = w->norm_running_var + p->hidden_sizes[2];
    ResNetStageV2(ctx, output, s->hidden3_0, s, w, p, p->hidden_sizes[2], p->hidden_sizes[3], -1, p->depths[3]);
    // for (int i = s->N * p->hidden_sizes[3] * s->H_out * s->W_out - 320; i < s->N * p->hidden_sizes[3] * s->H_out * s->W_out; i++) {
    // // for (int i = 0; i < 320; i++) {
    //     printf("%d=%f ", i, *(s->hidden3_1 + i));
    // }
}


void resnet_forward(Context *ctx, ResNet *model, Image* img, int B) {

    ResNetConfig* p = &model->config;
    p->num_channels = img->channel;
    p->downsample_in_bottleneck = 0;
    p->hidden_act = "relu";
    p->num_labels = 1000;
    RunState* s = &model->state;
    ResNetWeights *w = &model->weights;
    s->N = B;

    int max_mem = s->N * p->embedding_size * ((img->height + 2 * 3 - 7 + 1) / 2) * ((img->width  + 2 * 3 - 7 + 1 ) / 2) * sizeof(float);
    s->hidden0_0 = (float*)malloc(max_mem);
    s->hidden0_1 = (float*)malloc(max_mem);
    s->hidden1_0 = (float*)malloc(max_mem);
    s->hidden1_1 = (float*)malloc(max_mem);
    s->hidden1_2 = (float*)malloc(max_mem);
    s->hidden2_0 = (float*)malloc(max_mem);
    s->hidden2_1 = (float*)malloc(max_mem);
    s->hidden3_0 = (float*)malloc(max_mem);
    s->hidden3_1 = (float*)malloc(max_mem);

    s->xb = (float*)malloc(max_mem);
    s->xb1 = (float*)malloc(max_mem);
    s->xb2 = (float*)malloc(max_mem);
    s->xb3 = (float*)malloc(max_mem);
    s->xb4 = (int*)malloc(B);

    s->x = (float*)malloc(B * img->channel * img->height * img->width * sizeof(float));
    memcpy(s->x, img->data, B * img->channel * img->height * img->width * sizeof(float));

    s->H_in = img->height;
    s->W_in = img->width;
    w->conv_weight = model->weights.embedder;
    s->max_mem = s->N * p->num_channels * s->H_in * s->W_in;
    ResNetEmbeddingsV2(ctx, s->xb, s->x, s, w, p);
    // // for (int i = s->N * p->embedding_size * s->H_out * s->W_out - 320; i < s->N * p->embedding_size * s->H_out * s->W_out; i++) {
    // for (int i = 1 * p->embedding_size * s->H_out * s->W_out; i < 1 * p->embedding_size * s->H_out * s->W_out + 320; i++) {
    // // for (int i = 0; i < 320; i++) {
    //     printf("%d=%f ", i, *(s->xb + i));
    // }

    // resnet.encoder.stages.{i}.layers.0.shortcut
    w->conv_weight = w->norm_running_var + p->embedding_size;
    s->H_in = s->H_out;
    s->W_in = s->W_out;
    // ResNetShortCut(ctx, s->xb4, s->hidden1, s->hidden, s, w);
    int reduction = 4;
    // ResNetBottleNeckLayer(ctx, s->xb1, s->xb, s, w, reduction, 1);
    // ResNetStage(ctx, s->xb1, s->xb, s, w, 1);
    // ResNetStageV2(ctx, s->xb1, s->xb, s, w, p, p->embedding_size, p->hidden_sizes[0], 1, p->depths[0]);
    ResNetEncoderV2(ctx, s->xb1, s->xb, s, w, p);

    // for (int i = 0; i < s->N * p->hidden_sizes[3] * s->H_out * s->W_out; i++) {
    // // for (int i = 0; i < 320; i++) {
    //     if ((*(s->xb1 + i)) != 0.0f) {
    //         printf("%d=%f ", i, *(s->xb1 + i));
    //     }
    // }

    s->H_in = s->H_out;
    s->W_in = s->W_out;
    s->H_out = 1;
    s->W_out = 1;
    adaptiveavgpool2d_1x1_forward(s->xb2, s->xb1, s->N, p->hidden_sizes[3], s->H_in, s->W_in);
    // for (int i = 0; i < s->N * p->hidden_sizes[3]; i++) {
    //     printf("%d=%f ", i, *(s->xb2 + i));
    // }

    w->norm_weight = w->norm_running_var + p->hidden_sizes[3];
    w->norm_bias = w->norm_weight + p->hidden_sizes[3] * p->num_labels;
    linear_forward(s->xb3, s->xb2, w->norm_weight, w->norm_bias, s->N, p->num_labels, p->hidden_sizes[3]);
    // for (int i = 0; i < s->N * p->num_labels; i++) {
    //     printf("%d=%f ", i, *(s->xb3 + i));
    // }
    softmax_forward(s->xb3, s->N, p->num_labels);
    // for (int i = 0; i < s->N * p->num_labels; i++) {
    //     printf("%d=%f ", i, *(s->xb3 + i));
    // }
    argmax_forward(s->xb4, s->xb3, s->N, p->num_labels);
    for (int i = 0; i < s->N; i++) {
        printf("image:%d, label id: %d prob:%f \n", i, *(s->xb4 + i), s->xb3[*(s->xb4 + i)]);
    }
}

int main(int argc, char** argv) {
    ResNet model;
    resnet_build_from_checkpoint(&model, "microsoft-resnet-50.bin");

    Image img;
    read_image(&img, "image.bin");

    int B = 2;
    Context ctx;
    resnet_forward(&ctx, &model, &img, B);
    // gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
    return 0;
}