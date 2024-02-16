#ifndef KERNELS_CUH
#define KERNELS_CUH

#define BLOCK_SIZE_X 16 // Be careful, these
#define BLOCK_SIZE_Y 16 // are also defined in main.cu
#define TPB 256 // This one, too

// Prototypes of the Cuda kernels

__global__ void calc_gradient_mb_sum_gpu_w(float *weight_grad_inp, float *weight_grad, float *neuron_value,
                                           unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias, unsigned long long int *neighbour_number, unsigned long long int *graph_n, unsigned long long int *graph_i,
                                           unsigned long long int neuron_num, unsigned long long int all_input_num, unsigned long long int all_neighbour_num, unsigned long long int mini_batch_len);

__global__ void calc_gradient_mb_sum_gpu_b(float *weight_grad_inp, float *bias_grad,
                                           unsigned long long int neuron_num, unsigned long long int all_input_num, unsigned long long int all_neighbour_num, unsigned long long int mini_batch_len);

__global__ void calc_gradient_mb_sum_gpu(float *weight_grad_inp, float *weight_grad_temp,
                                         float *neuron_value,
                                         unsigned long long int mini_batch_len,
                                         unsigned long long int *neighbour_number,
                                         unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias,
                                         unsigned long long int *graph_n, unsigned long long int *graph_i,
                                         unsigned long long int neuron_num, unsigned long long int all_input_num, unsigned long long int all_neighbour_num);

__global__ void weight_transpose_gpu(unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias, unsigned long long int *first_ind_parent,
                                     unsigned long long int *neighbour_number, unsigned long long int *bias_number, unsigned long long int *parent_number,
                                     unsigned long long int *graph_p, unsigned long long int *graph_p_ind_n,
                                     float *weight, float *weight_trans,
                                     unsigned long long int neuron_num);

__global__ void add_bias_bcast(unsigned long long int mini_batch_len, unsigned long long int all_input_num,
                               float *bias, float *input_value);

__global__ void calc_grad_help_gpu(unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias, unsigned long long int *bias_number,
                                   float *input_value, float *weight_grad_help, float *weight_grad_help_temp, unsigned long long int *activation_type,
                                   unsigned long long int neuron_num, unsigned long long int all_input_num, unsigned long long int mini_batch_len);

__global__ void calc_neuron_mb_gpu(unsigned long long int *bias_number,
                                   unsigned long long int *first_ind_bias, unsigned long long int *activation_type, float *bias,
                                   float *neuron_value, float *input_value,
                                   unsigned long long int neuron_num, unsigned long long int all_input_num, unsigned long long int mini_batch_len);

__global__ void calc_grad_help_0_gpu(unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias, unsigned long long int *bias_number,
                                     float *input_value, float *weight_grad_help_temp, unsigned long long int *activation_type,
                                     unsigned long long int neuron_num, unsigned long long int all_input_num, unsigned long long int mini_batch_len);

__global__ void update_weight_gd_h_gpu(float *weight, float *bias,
                                     float *weight_grad, float *bias_grad,
                                     float *vt_weight, float *vt_bias,
                                     unsigned long long int *graph_logic,
                                     unsigned long long int *bias_logic,
                                     unsigned long long int all_input_num, unsigned long long int all_neighbour_num,
                                     float grad_alpha, float adam_beta1);

__global__ void update_bias_gd_h_gpu(float *weight, float *bias,
                                   float *weight_grad, float *bias_grad,
                                   float *vt_weight, float *vt_bias,
                                   unsigned long long int *graph_logic,
                                   unsigned long long int *bias_logic,
                                   unsigned long long int all_input_num, unsigned long long int all_neighbour_num,
                                   float grad_alpha, float adam_beta1);

__global__ void update_weight_gd_gpu(float *weight, float *bias,
                                     float *weight_grad, float *bias_grad,
                                     float *vt_weight, float *vt_bias,
                                     unsigned long long int *graph_logic,
                                     unsigned long long int *bias_logic,
                                     unsigned long long int all_input_num, unsigned long long int all_neighbour_num,
                                     float grad_alpha, float adam_beta1, unsigned long long int nesterov);

__global__ void update_bias_gd_gpu(float *weight, float *bias,
                                   float *weight_grad, float *bias_grad,
                                   float *vt_weight, float *vt_bias,
                                   unsigned long long int *graph_logic,
                                   unsigned long long int *bias_logic,
                                   unsigned long long int all_input_num, unsigned long long int all_neighbour_num,
                                   float grad_alpha, float adam_beta1, unsigned long long int nesterov);

__global__ void update_weight_adam_gpu(float *weight, float *bias,
                                       float *weight_grad, float *bias_grad,
                                       float *mt_weight, float *mt_bias,
                                       float *vt_weight, float *vt_bias,
                                       float *mth_weight, float *mth_bias,
                                       float *vth_weight, float *vth_bias,
                                       unsigned long long int *graph_logic,
                                       unsigned long long int *bias_logic,
                                       unsigned long long int all_input_num, unsigned long long int all_neighbour_num,
                                       float adam_alpha, float adam_beta1, float adam_beta2,
                                       float adam_beta1t, float adam_beta2t, float adam_eps);

__global__ void update_bias_adam_gpu(float *weight, float *bias,
                                     float *weight_grad, float *bias_grad,
                                     float *mt_weight, float *mt_bias,
                                     float *vt_weight, float *vt_bias,
                                     float *mth_weight, float *mth_bias,
                                     float *vth_weight, float *vth_bias,
                                     unsigned long long int *graph_logic,
                                     unsigned long long int *bias_logic,
                                     unsigned long long int all_input_num, unsigned long long int all_neighbour_num,
                                     float adam_alpha, float adam_beta1, float adam_beta2,
                                     float adam_beta1t, float adam_beta2t, float adam_eps);

__global__ void update_weight_adamax_gpu(float *weight, float *bias,
                                       float *weight_grad, float *bias_grad,
                                       float *mt_weight, float *mt_bias,
                                       float *ut_weight, float *ut_bias,
                                       unsigned long long int *graph_logic,
                                       unsigned long long int *bias_logic,
                                       unsigned long long int all_input_num, unsigned long long int all_neighbour_num,
                                       float adam_alpha, float adam_beta1, float adam_beta2,
                                       float adam_beta1t, float adam_beta2t, float adam_eps);

__global__ void update_bias_adamax_gpu(float *weight, float *bias,
                                     float *weight_grad, float *bias_grad,
                                     float *mt_weight, float *mt_bias,
                                     float *ut_weight, float *ut_bias,
                                     unsigned long long int *graph_logic,
                                     unsigned long long int *bias_logic,
                                     unsigned long long int all_input_num, unsigned long long int all_neighbour_num,
                                     float adam_alpha, float adam_beta1, float adam_beta2,
                                     float adam_beta1t, float adam_beta2t, float adam_eps);


__global__ void copy_input_gpu(float *datas, float *input_value,
                               unsigned long long int *first_ind_bias,
                               unsigned long long int mini_batch_len,
                               unsigned long long int all_input_num, unsigned long long int neuron_num,
                               unsigned long long int input_num, unsigned long long int output_num);

__global__ void set_zero_gpu(float *v, unsigned long long int N);

__device__ float act_fun_gpu(float y, unsigned long long int act_type);

__device__ float act_fun_diff_gpu(float x, unsigned long long int chooser);

__device__ __forceinline__ float atomicMaxf(float *addr, float value);

__global__ void maxnorm(const float *__restrict__ input, const unsigned long long int size, float *maxOut);

__global__ void maxnormDiff(const float *__restrict__ input1, const float *__restrict__ input2, const unsigned long long int size, float *maxOut);

__global__ void calc_gradient_mb_gpu(float *weight_grad_inp, float *weight_grad_inp_temp, float *weight_grad_inp_old,
                                     float *weight_grad_help,
                                     unsigned long long int mini_batch_len,
                                     unsigned long long int *neighbour_number, unsigned long long int *bias_number,
                                     unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias,
                                     unsigned long long int *graph_n, unsigned long long int *graph_i,
                                     float *weight,
                                     unsigned long long int neuron_num, unsigned long long int all_input_num);

__global__ void calc_network_mb_gpu(float *datas, unsigned long long int mini_batch_len,
                                    unsigned long long int *bias_number, unsigned long long int *parent_number,
                                    unsigned long long int *first_ind_bias, unsigned long long int *first_ind_parent,
                                    unsigned long long int *graph_p,
                                    float *weight_trans, float *neuron_value, float *input_value,
                                    unsigned long long int neuron_num, unsigned long long int all_input_num);

__global__ void calc_network_mb_gpu_ff(float *datas, unsigned long long int mini_batch_len,
                                       unsigned long long int *bias_number, unsigned long long int *parent_number,
                                       unsigned long long int *first_ind_bias, unsigned long long int *first_ind_parent,
                                       unsigned long long int *graph_p,
                                       float *weight_trans, float *neuron_value, float *input_value,
                                       unsigned long long int neuron_num, unsigned long long int all_input_num,
                                       unsigned long long int act_dist, unsigned long long int *dist_input);

__global__ void calc_neuron_mb_gpu_ff(unsigned long long int *bias_number,
                                      unsigned long long int *first_ind_bias, unsigned long long int *activation_type, float *bias,
                                      float *neuron_value, float *input_value,
                                      unsigned long long int neuron_num, unsigned long long int all_input_num, unsigned long long int mini_batch_len,
                                      unsigned long long int act_dist, unsigned long long int *dist);

__global__ void calc_gradient_mb_gpu_ff(float *weight_grad_inp, float *weight_grad_inp_old,
                                        float *weight_grad_help,
                                        unsigned long long int mini_batch_len,
                                        unsigned long long int *neighbour_number, unsigned long long int *bias_number,
                                        unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias,
                                        unsigned long long int *graph_n, unsigned long long int *graph_i,
                                        float *weight,
                                        unsigned long long int neuron_num, unsigned long long int all_input_num,
                                        unsigned long long int act_layer, unsigned long long int *dist);

__global__ void divide_gpu(float *v, unsigned long long int N, unsigned long long int number);

__global__ void l1norm(const float *__restrict__ input, const unsigned long long int size, float *sumOut);

__global__ void l1normdiff(const float *__restrict__ input1, const float *__restrict__ input2, const unsigned long long int size, float *sumOut);

__device__ float my_atomicAdd(float* addr, float val);

__global__ void reg_weight_gpu(float *weight, float *bias,
                               float *weight_grad, float *bias_grad,
                               unsigned long long int all_input_num, unsigned long long int all_neighbour_num,
                               float alpha);

__global__ void reg_bias_gpu(float *weight, float *bias,
                               float *weight_grad, float *bias_grad,
                               unsigned long long int all_input_num, unsigned long long int all_neighbour_num,
                               float alpha);

__global__ void clipping_weight_gpu(float *weight_grad, float *bias_grad,
                                    unsigned long long int all_input_num, unsigned long long int all_neighbour_num,
                                    float threshold);

__global__ void clipping_bias_gpu(float *weight_grad, float *bias_grad,
                                    unsigned long long int all_input_num, unsigned long long int all_neighbour_num,
                                    float threshold);

#endif