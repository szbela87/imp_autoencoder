/**
 *
 *  Generalized Neural Network
 *
 *      by: Béla J. Szekeres, PhD in applied mathematics
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <omp.h>
//#include <ncurses.h>

// Cuda kernels
#include "kernels.cuh"

/* Global variables */
unsigned long long int seed;
unsigned long long int lr_scheduler;
unsigned long long int thread_num;
unsigned long long int shuffle_num;
float tol_gradnorm;
float tol_error_diff;
float tol_fixit;
unsigned long long int maxiter_grad;
unsigned long long int range_div;
unsigned long long int maxiter_fix;
float initdx;
unsigned long long int sfreq;
char input_name[100];
char input_name_valid[100];
char input_name_test[100];
char output_name[100];
char predict_name_valid[100];
char predict_name_test[100];
char test_log[100];
char test_log_final[100];
unsigned long long int learn_num;
unsigned long long int valid_num;
unsigned long long int test_num;
unsigned long long int mini_batch_size;
unsigned long long int neuron_num;
unsigned long long int input_num;
unsigned long long int output_num;
char graph_datas[100];
char logic_datas[100];
char fixwb_datas[100];
char train_lossfunction_type[100];
char valid_metric_type[100];
char valid_metric_type_2[100];
float alpha;
unsigned long long int optimizer;
float grad_alpha;
float adam_alpha;
float adam_beta1;
float adam_beta2;
float adam_eps;
unsigned long long int early_stopping;
unsigned long long int ff_optimization;
unsigned long long int clipping;
float clipping_threshold;
unsigned long long int loaddatas;
char load_backup[100];
char save_best_model[100];
unsigned long long int numgrad;
float numgrad_eps;
float inf;
unsigned long long int all_neighbour_num;
unsigned long long int all_input_num;
unsigned long long int max_bias_num;
unsigned long long int shared_weights_num;
unsigned long long int shared_biases_num;
char shared_w_datas[100];
char shared_b_datas[100];
unsigned long long int nesterov;
unsigned long long int cyclic_momentum;
float base_momentum;
float max_momentum;
float div_factor;
float final_div_factor;
float pcr;
unsigned long long int step_size;
float lr_gamma;

float valid_std, valid_mean, valid_std_best, valid_mean_best, valid_th, valid_th_best, valid_metric;

/* Prototypes */
float f_lr_momentum(unsigned long long int i, float start, float end, unsigned long long int T);
void read_parameters(char file_name[100]);
void read_data(float *datas, unsigned long long int line_number, FILE *f_data, unsigned long long int test);
void read_graph(char graph_file_name[100], char logic_file_name[100], char fixwb_file_name[100],
                char shared_w_file_name[100], char shared_b_file_name[100],
                unsigned long long int *neighbour_number, unsigned long long int *bias_number,
                unsigned long long int *shared_weights_blocksize, unsigned long long int *shared_biases_blocksize,
                unsigned long long int **shared_weights_indices,
                unsigned long long int **shared_weights, unsigned long long int **shared_biases,
                unsigned long long int **activation_type, unsigned long long int **graph_n, unsigned long long int **graph_i,
                unsigned long long int **graph_logic, unsigned long long int **bias_logic, unsigned long long int **parent_number,
                float **fix_weight, float **fix_bias,
                unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias);
void predict(float *train_data, float *valid_data, float *test_data,
             unsigned long long int mini_batch_num, unsigned long long int mini_batch_num_valid, unsigned long long int mini_batch_num_test,
             unsigned long long int *neighbour_number_g, unsigned long long int *neighbour_number, unsigned long long int *bias_number_g, unsigned long long int *bias_number, unsigned long long int *parent_number_g,
             unsigned long long int *first_ind_neighbour_g, unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias_g, unsigned long long int *first_ind_bias, unsigned long long int *first_ind_parent_g,
             unsigned long long int *activation_type_g, unsigned long long int *activation_type, unsigned long long int *graph_n_g, unsigned long long int *graph_n, unsigned long long int *graph_i_g, unsigned long long int *graph_i, unsigned long long int *graph_p_g,
             unsigned long long int *graph_p_ind_n_g,
             float *weight_g, float *weight_trans_g, float *bias_g,
             unsigned long long int dist_max,
             unsigned long long int *dist_g, unsigned long long int *dist,
             unsigned long long int *dist_input_g, unsigned long long int *dist_input,
             float **predictions_valid, float **predictions_test, float *errors_valid, float *errors_test);

unsigned long long int rand_range_int(unsigned long long int min, unsigned long long int max);
float rand_range(float min, float max);

float act_fun(float x, unsigned long long int chooser);
float act_fun_diff(float x, unsigned long long int chooser);

unsigned long long int imax(unsigned long long int a, unsigned long long int b);
float dmax(float a, float b);
float **allocate_dmatrix(unsigned long long int row_num, unsigned long long int *col_num);
unsigned long long int **allocate_imatrix(unsigned long long int row_num, unsigned long long int *col_num);
void deallocate_dmatrix(float **m, unsigned long long int row_num);
void deallocate_imatrix(unsigned long long int **m, unsigned long long int row_num);
void print_progress_bar(unsigned long long int max_length, float rate);
float calc_error(float *neuron_value, float *target_vector, unsigned long long int mini_batch_len);

void print_graph(unsigned long long int *neighbour_number, unsigned long long int *bias_number, unsigned long long int *activation_type, unsigned long long int *graph_n, unsigned long long int *graph_i,
                 unsigned long long int *graph_logic, unsigned long long int *bias_logic,
                 float *fix_weight, float *fix_bias,
                 unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias);

void program_failure(char str[]);
float random_normal(float mean, float std_dev);
void softmax(float *input, unsigned long long int input_len);

void copy_dmatrix(float **input_matrix, unsigned long long int row_num, unsigned long long int *col_num, float *output_matrix);
void copy_imatrix(unsigned long long int **input_matrix, unsigned long long int row_num, unsigned long long int *col_num, unsigned long long int *output_matrix);

void initialize_weights(unsigned long long int *neighbour_number, unsigned long long int *bias_number,
                        unsigned long long int *shared_weights_blocksize, unsigned long long int *shared_biases_blocksize,
                        unsigned long long int **shared_weights_indices,
                        unsigned long long int **shared_weights, unsigned long long int **shared_biases,
                        unsigned long long int **activation_type, unsigned long long int **graph_n, unsigned long long int **graph_i,
                        unsigned long long int **parent_number, float **weight, float **bias);

float calc_gradient_mini_batch(float *datas, unsigned long long int mini_batch_len,
                               unsigned long long int *neighbour_number_g, unsigned long long int *neighbour_number, unsigned long long int *bias_number_g, unsigned long long int *bias_number, unsigned long long int *parent_number_g,
                               unsigned long long int *first_ind_neighbour_g, unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias_g, unsigned long long int *first_ind_bias, unsigned long long int *first_ind_parent_g,
                               unsigned long long int *activation_type_g, unsigned long long int *activation_type, unsigned long long int *graph_n_g, unsigned long long int *graph_n, unsigned long long int *graph_i_g, unsigned long long int *graph_i, unsigned long long int *graph_p_g,
                               unsigned long long int *graph_p_ind_n_g,
                               float *weight_g, float *bias_g,
                               float *weight_grad_g, float *bias_grad_g,
                               float *iter_forward, float *iter_backward);

float calc_gradient_mini_batch_ff(float *datas, unsigned long long int mini_batch_len,
                                  unsigned long long int *neighbour_number_g, unsigned long long int *neighbour_number, unsigned long long int *bias_number_g, unsigned long long int *bias_number, unsigned long long int *parent_number_g,
                                  unsigned long long int *first_ind_neighbour_g, unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias_g, unsigned long long int *first_ind_bias, unsigned long long int *first_ind_parent_g,
                                  unsigned long long int *activation_type_g, unsigned long long int *activation_type, unsigned long long int *graph_n_g, unsigned long long int *graph_n, unsigned long long int *graph_i_g, unsigned long long int *graph_i, unsigned long long int *graph_p_g,
                                  unsigned long long int *graph_p_ind_n_g,
                                  float *weight_g, float *bias_g,
                                  float *weight_grad_g, float *bias_grad_g,
                                  float *iter_forward, float *iter_backward,
                                  unsigned long long int dist_max,
                                  unsigned long long int *dist_g, unsigned long long int *dist,
                                  unsigned long long int *dist_input_g, unsigned long long int *dist_input);

float calc_network_mini_batch(float *datas, unsigned long long int mini_batch_len,
                              unsigned long long int *neighbour_number_g, unsigned long long int *neighbour_number, unsigned long long int *bias_number_g, unsigned long long int *bias_number, unsigned long long int *parent_number_g,
                              unsigned long long int *first_ind_neighbour_g, unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias_g, unsigned long long int *first_ind_bias, unsigned long long int *first_ind_parent_g,
                              unsigned long long int *activation_type_g, unsigned long long int *activation_type, unsigned long long int *graph_n_g, unsigned long long int *graph_n, unsigned long long int *graph_i_g, unsigned long long int *graph_i, unsigned long long int *graph_p_g,
                              unsigned long long int *graph_p_ind_n_g,
                              float *weight_trans_g, float *bias_g,
                              float *iter_forward);

float calc_network_mini_batch_ff(float *datas, unsigned long long int mini_batch_len,
                                 unsigned long long int *neighbour_number_g, unsigned long long int *neighbour_number, unsigned long long int *bias_number_g, unsigned long long int *bias_number, unsigned long long int *parent_number_g,
                                 unsigned long long int *first_ind_neighbour_g, unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias_g, unsigned long long int *first_ind_bias, unsigned long long int *first_ind_parent_g,
                                 unsigned long long int *activation_type_g, unsigned long long int *activation_type, unsigned long long int *graph_n_g, unsigned long long int *graph_n, unsigned long long int *graph_i_g, unsigned long long int *graph_i, unsigned long long int *graph_p_g,
                                 unsigned long long int *graph_p_ind_n_g,
                                 float *weight_trans_g, float *bias_g,
                                 float *iter_forward,
                                 unsigned long long int dist_max,
                                 unsigned long long int *dist_g, unsigned long long int *dist,
                                 unsigned long long int *dist_input_g, unsigned long long int *dist_input);

void make_predictions(float *datas, unsigned long long int mini_batch_len,
                      unsigned long long int *neighbour_number_g, unsigned long long int *neighbour_number, unsigned long long int *bias_number_g, unsigned long long int *bias_number, unsigned long long int *parent_number_g,
                      unsigned long long int *first_ind_neighbour_g, unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias_g, unsigned long long int *first_ind_bias, unsigned long long int *first_ind_parent_g,
                      unsigned long long int *activation_type_g, unsigned long long int *activation_type, unsigned long long int *graph_n_g, unsigned long long int *graph_n, unsigned long long int *graph_i_g, unsigned long long int *graph_i, unsigned long long int *graph_p_g,
                      unsigned long long int *graph_p_ind_n_g,
                      float *weight_trans_g, float *bias_g,
                      float **predictions_mini_batch);

void make_predictions_ff(float *datas, unsigned long long int mini_batch_len,
                         unsigned long long int *neighbour_number_g, unsigned long long int *neighbour_number, unsigned long long int *bias_number_g, unsigned long long int *bias_number, unsigned long long int *parent_number_g,
                         unsigned long long int *first_ind_neighbour_g, unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias_g, unsigned long long int *first_ind_bias, unsigned long long int *first_ind_parent_g,
                         unsigned long long int *activation_type_g, unsigned long long int *activation_type, unsigned long long int *graph_n_g, unsigned long long int *graph_n, unsigned long long int *graph_i_g, unsigned long long int *graph_i, unsigned long long int *graph_p_g,
                         unsigned long long int *graph_p_ind_n_g,
                         float *weight_trans_g, float *bias_g,
                         unsigned long long int dist_max,
                         unsigned long long int *dist_g, unsigned long long int *dist,
                         unsigned long long int *dist_input_g, unsigned long long int *dist_input,
                         float **predictions_mini_batch);

void save_weight_bias(char filename[100], float *weight, float *bias,
                      unsigned long long int neuron_num, unsigned long long int *neighbour_number, unsigned long long int *bias_number,
                      unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias);

void load_weight_bias(char filename[100], float *weight, float *bias,
                      unsigned long long int neuron_num, unsigned long long int *neighbour_number, unsigned long long int *bias_number,
                      unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias);
float calc_vector_norm(float *v1, unsigned long long int row_nums);
float calc_vector_max(float *v1, unsigned long long int row_nums);
float calc_diff_vectors(float *v1, float *v2, unsigned long long int row_nums);
float trapz(float *x, float *y, unsigned long long int x_size);
float calculate_auc(float *fpr, float *tpr, unsigned long long int fpr_size);
void calculate_tpr_fpr_bc(unsigned long long int *predictions, unsigned long long int *targets, unsigned long long int length, float *tpr, float *fpr, float *f1score, float *accuracy, float *precision);
float calculate_mcc(unsigned long long int *predictions, unsigned long long int *targets, unsigned long long int length);

int main()
{
    cudaError_t cudaStatus;

    // Initialize the CUDA device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    // Allocatables

    // Set inf
    inf = 1.e20;
    // initscr();

    // Read input parameters
    read_parameters("./inputs/simulparams.dat");

    // Set number of threads
    omp_set_num_threads(thread_num);

    // Set random seed
    if (seed == 0)
    {
        srand(time(0));
    }
    else
    {
        srand(seed);
    }

    cudaError_t error;

    cudaEvent_t start, stop;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float **predictions_valid = (float **)malloc(valid_num * sizeof(float *));
    for (unsigned long long int i = 0; i < valid_num; i++)
    {
        predictions_valid[i] = (float *)malloc(output_num * sizeof(float));
    }
    float **predictions_test = (float **)malloc(test_num * sizeof(float *));
    for (unsigned long long int i = 0; i < test_num; i++)
    {
        predictions_test[i] = (float *)malloc(output_num * sizeof(float));
    }

    float *errors_valid = (float *)malloc(valid_num * sizeof(float));
    float *errors_test = (float *)malloc(test_num * sizeof(float));

    // Minibatch allocations
    unsigned long long int mini_batch_num = learn_num / mini_batch_size;
    if (mini_batch_size * mini_batch_num != learn_num)
    {
        mini_batch_num++;
    }
    unsigned long long int mini_batch_num_valid = valid_num / mini_batch_size;
    if (mini_batch_size * mini_batch_num_valid != valid_num)
    {
        mini_batch_num_valid++;
    }
    sfreq = mini_batch_num / sfreq;
    unsigned long long int mini_batch_num_test = test_num / mini_batch_size;
    if (mini_batch_size * mini_batch_num_test != test_num)
    {
        mini_batch_num_test++;
    }

    float *datas_mini_batch = (float *)malloc(mini_batch_size * (input_num + output_num) * sizeof(float));
    float *train_data = (float *)malloc(learn_num * (input_num + output_num) * sizeof(float));
    float *valid_data = (float *)malloc(valid_num * (input_num + output_num) * sizeof(float));
    float *test_data = (float *)malloc(test_num * (input_num + output_num) * sizeof(float));

    unsigned long long int *test_pred_labels = (unsigned long long int *)malloc(test_num * sizeof(unsigned long long int));
    unsigned long long int *test_labels = (unsigned long long int *)malloc(test_num * sizeof(unsigned long long int));

    unsigned long long int *valid_labels = (unsigned long long int *)malloc(valid_num * sizeof(unsigned long long int));

    float *valid_fpr = (float *)malloc((range_div + 1) * sizeof(float));
    float *valid_tpr = (float *)malloc((range_div + 1) * sizeof(float));
    float *mcc_list = (float *)malloc((range_div + 1) * sizeof(float));
    float *f1score_list = (float *)malloc((range_div + 1) * sizeof(float));

    // Allocations for the graph
    unsigned long long int *neighbour_number = (unsigned long long int *)malloc(neuron_num * sizeof(unsigned long long int));
    unsigned long long int *bias_number = (unsigned long long int *)malloc(neuron_num * sizeof(unsigned long long int));
    unsigned long long int *first_ind_neighbour = (unsigned long long int *)malloc(neuron_num * sizeof(unsigned long long int));
    unsigned long long int *first_ind_bias = (unsigned long long int *)malloc(neuron_num * sizeof(unsigned long long int));

    for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        first_ind_neighbour[neuron_id] = 0;
        first_ind_bias[neuron_id] = 0;
    }

    unsigned long long int **activation_type_m = (unsigned long long int **)malloc(neuron_num * sizeof(unsigned long long int *));
    unsigned long long int **graph_n_m = (unsigned long long int **)malloc(neuron_num * sizeof(unsigned long long int *));
    unsigned long long int **graph_i_m = (unsigned long long int **)malloc(neuron_num * sizeof(unsigned long long int *));
    unsigned long long int **graph_logic_m = (unsigned long long int **)malloc(neuron_num * sizeof(unsigned long long int *));
    unsigned long long int **bias_logic_m = (unsigned long long int **)malloc(neuron_num * sizeof(unsigned long long int *));
    float **fix_weight_m = (float **)malloc(neuron_num * sizeof(float *));
    float **fix_bias_m = (float **)malloc(neuron_num * sizeof(float *));
    unsigned long long int **parent_number_m = (unsigned long long int **)malloc(neuron_num * sizeof(unsigned long long int *));

    // Allocations for shared weights and biases
    unsigned long long int *shared_weights_blocksize, *shared_biases_blocksize, **shared_weights, **shared_biases,
        **shared_weights_indices;
    float *shared_weights_values, *shared_biases_values;

    if (shared_weights_num > 0)
    {
        shared_weights_blocksize = (unsigned long long int *)calloc(shared_weights_num, sizeof(unsigned long long int));
        shared_weights_values = (float *)calloc(shared_weights_num, sizeof(float));
        shared_weights = (unsigned long long int **)calloc(shared_weights_num, sizeof(unsigned long long int *));
        shared_weights_indices = (unsigned long long int **)calloc(shared_weights_num, sizeof(unsigned long long int *));
    }

    if (shared_biases_num > 0)
    {
        shared_biases_blocksize = (unsigned long long int *)calloc(shared_biases_num, sizeof(unsigned long long int));
        shared_biases_values = (float *)calloc(shared_biases_num, sizeof(float));
        shared_biases = (unsigned long long int **)calloc(shared_biases_num, sizeof(unsigned long long int *));
    }

    read_graph(graph_datas, logic_datas, fixwb_datas,
               shared_w_datas, shared_b_datas,
               neighbour_number, bias_number,
               shared_weights_blocksize, shared_biases_blocksize,
               shared_weights_indices,
               shared_weights, shared_biases,
               activation_type_m, graph_n_m, graph_i_m, graph_logic_m, bias_logic_m, parent_number_m,
               fix_weight_m, fix_bias_m,
               first_ind_neighbour, first_ind_bias);

    unsigned long long int *first_ind_parent = (unsigned long long int *)calloc(all_input_num, sizeof(unsigned long long int));
    unsigned long long int parent_number_old = 0;
    unsigned long long int ind = 0;

    // first parent indices in all_input_num sized vectors
    for (unsigned long long int i = 0; i < neuron_num; i++)
    {
        for (unsigned long long int j = 0; j < bias_number[i]; j++)
        {
            if (ind > 0)
            {
                first_ind_parent[ind] = first_ind_parent[ind - 1] + parent_number_old;
            }
            parent_number_old = parent_number_m[i][j];
            ind++;
        }
    }

    max_bias_num = 0;
    for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        if (bias_number[neuron_id] > max_bias_num)
        {
            max_bias_num = bias_number[neuron_id];
        }
    }

    unsigned long long int *activation_type = (unsigned long long int *)malloc(all_input_num * sizeof(unsigned long long int));
    unsigned long long int *graph_n = (unsigned long long int *)malloc(all_neighbour_num * sizeof(unsigned long long int));
    unsigned long long int *graph_i = (unsigned long long int *)malloc(all_neighbour_num * sizeof(unsigned long long int));
    unsigned long long int *graph_logic = (unsigned long long int *)malloc(all_neighbour_num * sizeof(unsigned long long int));
    unsigned long long int *bias_logic = (unsigned long long int *)malloc(all_input_num * sizeof(unsigned long long int));
    float *fix_weight = (float *)malloc(all_neighbour_num * sizeof(float));
    float *fix_bias = (float *)malloc(all_input_num * sizeof(float));
    unsigned long long int *parent_number = (unsigned long long int *)malloc(all_input_num * sizeof(unsigned long long int));

    // Flattening
    copy_imatrix(activation_type_m, neuron_num, bias_number, activation_type);
    copy_imatrix(graph_n_m, neuron_num, neighbour_number, graph_n);
    copy_imatrix(graph_i_m, neuron_num, neighbour_number, graph_i);
    copy_imatrix(graph_logic_m, neuron_num, neighbour_number, graph_logic);
    copy_imatrix(bias_logic_m, neuron_num, bias_number, bias_logic);
    copy_imatrix(parent_number_m, neuron_num, bias_number, parent_number);
    copy_dmatrix(fix_weight_m, neuron_num, neighbour_number, fix_weight);
    copy_dmatrix(fix_bias_m, neuron_num, bias_number, fix_bias);

    // Copying parent_number_m to parent_number??

    // Creating the reversed graph, the second array is for the neighbor indices (the order, not the index)
    unsigned long long int **graph_p_m = (unsigned long long int **)malloc(all_input_num * sizeof(unsigned long long int *));
    unsigned long long int **graph_p_ind_n_m = (unsigned long long int **)malloc(all_input_num * sizeof(unsigned long long int *));
    // Allocating it
    for (unsigned long long int i = 0; i < all_input_num; i++)
    {
        graph_p_m[i] = (unsigned long long int *)malloc(parent_number[i] * sizeof(unsigned long long int));
        graph_p_ind_n_m[i] = (unsigned long long int *)malloc(parent_number[i] * sizeof(unsigned long long int));
    }
    // Upload it
    unsigned long long int *graph_p_m_counter = (unsigned long long int *)calloc(all_input_num, sizeof(unsigned long long int));
    for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long long int neighbour_counter = 0; neighbour_counter < neighbour_number[neuron_id]; neighbour_counter++)
        {
            unsigned long long int neighbour_ind = graph_n_m[neuron_id][neighbour_counter];
            unsigned long long int bias_ind = graph_i_m[neuron_id][neighbour_counter];
            unsigned long long int input_ind = first_ind_bias[neighbour_ind] + bias_ind;
            graph_p_m[input_ind][graph_p_m_counter[input_ind]] = neuron_id; // we need a counter here
            graph_p_ind_n_m[input_ind][graph_p_m_counter[input_ind]] = neighbour_counter;
            graph_p_m_counter[input_ind]++;
        }
    }

    unsigned long long int *graph_p = (unsigned long long int *)malloc(all_neighbour_num * sizeof(unsigned long long int));
    unsigned long long int *graph_p_ind_n = (unsigned long long int *)malloc(all_neighbour_num * sizeof(unsigned long long int));
    // Flattening it
    copy_imatrix(graph_p_m, all_input_num, parent_number, graph_p);
    copy_imatrix(graph_p_ind_n_m, all_input_num, parent_number, graph_p_ind_n);

    // Initializing the network
    float **weight_m = (float **)malloc(neuron_num * sizeof(float *));
    float **bias_m = (float **)malloc(neuron_num * sizeof(float *));
    initialize_weights(neighbour_number, bias_number,
                       shared_weights_blocksize, shared_biases_blocksize,
                       shared_weights_indices,
                       shared_weights, shared_biases,
                       activation_type_m, graph_n_m, graph_i_m, parent_number_m, weight_m, bias_m);

    float *weight = (float *)malloc(all_neighbour_num * sizeof(float));
    float *bias = (float *)malloc(all_input_num * sizeof(float));
    copy_dmatrix(weight_m, neuron_num, neighbour_number, weight);
    copy_dmatrix(bias_m, neuron_num, bias_number, bias);

    float *vt_weight = (float *)calloc(all_neighbour_num, sizeof(float));
    float *vt_bias = (float *)calloc(all_input_num, sizeof(float));
    float *ut_weight = (float *)calloc(all_neighbour_num, sizeof(float));
    float *ut_bias = (float *)calloc(all_input_num, sizeof(float));
    float *mt_weight = (float *)calloc(all_neighbour_num, sizeof(float));
    float *mt_bias = (float *)calloc(all_input_num, sizeof(float));
    float *vth_weight = (float *)calloc(all_neighbour_num, sizeof(float));
    float *vth_bias = (float *)calloc(all_input_num, sizeof(float));
    float *mth_weight = (float *)calloc(all_neighbour_num, sizeof(float));
    float *mth_bias = (float *)calloc(all_input_num, sizeof(float));
    float *weight_grad = (float *)calloc(all_neighbour_num, sizeof(float));
    float *bias_grad = (float *)calloc(all_input_num, sizeof(float));

    float *weight_best = (float *)calloc(all_neighbour_num, sizeof(float));
    float *bias_best = (float *)calloc(all_input_num, sizeof(float));

    // Setting the fix weights and biases
    for (unsigned long long int ind = 0; ind < all_neighbour_num; ind++)
    {
        if (graph_logic[ind] == 0)
        {
            weight[ind] = fix_weight[ind];
        }
    }
    for (unsigned long long int ind = 0; ind < all_input_num; ind++)
    {
        if (bias_logic[ind] == 0)
        {
            bias[ind] = fix_bias[ind];
        }
    }

    // Loading the weights and biases
    float adam_beta1t = 1.0, adam_beta2t = 1.0;

    if (loaddatas == 1)
    {
        load_weight_bias(load_backup, weight, bias, neuron_num, neighbour_number, bias_number,
                         first_ind_neighbour, first_ind_bias);

        // Setting the fix weights and biases
        for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            unsigned long long int startind = first_ind_neighbour[neuron_id];
            for (unsigned long long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {

                if (graph_logic[startind + neighbour_ind] == 0)
                {
                    weight[startind + neighbour_ind] = fix_weight[startind + neighbour_ind];
                }
            }
        }

        for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            unsigned long long int startind = first_ind_bias[neuron_id];
            for (unsigned long long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {

                if (bias_logic[startind + bias_ind] == 0)
                {
                    bias[startind + bias_ind] = fix_bias[startind + bias_ind];
                }
            }
        }
    }

    //+++++++++++++++++++++++++//
    //                         //
    //      Input checking     //
    //                         //
    //+++++++++++++++++++++++++//
    if ((strcmp(train_lossfunction_type, "bce_multilabeling") != 0) && (strcmp(train_lossfunction_type, "multilabeling_crossentropy") != 0) && (strcmp(train_lossfunction_type, "multiclassification_crossentropy") != 0) && (strcmp(train_lossfunction_type, "MSE") != 0) && (strcmp(train_lossfunction_type, "MAE") != 0))
    {
        printf("The train_lossfunction_type should be:\n - 'multilabeling_crossentropy' or \n - 'multiclassification_crossentropy' or \n - 'MSE'\n - 'MAE'\n - 'bce_multilabeling'\n\n");
        program_failure("Wrong loss function");
    }

    if (strcmp(train_lossfunction_type, "bce_multilabeling") == 0)
    {
        // Checking that the output activations are identity, and all outputs have just one input.
        unsigned long long int output_logic = 0;
        for (unsigned long long int i = 0; i < output_num; i++)
        {
            unsigned long long int neuron_id = neuron_num - output_num + i;
            if (bias_number[neuron_id] != 1)
            {
                output_logic += 1;
            }
            if (neighbour_number[neuron_id] != 0)
            {
                output_logic += 1;
            }
            if (activation_type_m[neuron_id][0] != 0)
            {
                output_logic += 1;
            }
        }

        if (output_logic > 0)
        {
            program_failure("Wrong activation function type on the output!");
        }
    }

    // printf("\n vmb: %.5f vstd: %.5f  vth: %.5f \n", valid_mean_best, valid_std_best, valid_th_best);

    //+++++++++++++++++++++++++//
    //                         //
    //      PERT method        //
    //                         //
    //+++++++++++++++++++++++++//

    unsigned long long int check_cycle = 0;
    unsigned long long int *dist = (unsigned long long int *)malloc(neuron_num * sizeof(unsigned long long int));          // neuron distances for PERT method
    unsigned long long int *dist_input = (unsigned long long int *)malloc(all_input_num * sizeof(unsigned long long int)); // input distances for PERT method
    unsigned long long int *dist_extra = (unsigned long long int *)malloc(neuron_num * sizeof(unsigned long long int));    // neuron distances for PERT method
    unsigned long long int dist_max;
    if (ff_optimization > 0)
    {
        float et_ff = 0.0;
        cudaEventRecord(start, NULL);
        unsigned long long int output_num_temp = 0;

        // Calculate the distances for PERT method
        for (unsigned long long int i = 0; i < neuron_num; i++)
        {
            dist[i] = 0;
        }

        for (unsigned long long int i = 0; i < neuron_num; i++)
        {
            for (unsigned long long int j = 0; j < 400; j++)
            {
                printf("\b \b");
            }
            printf("\r");
            printf("%.35s", "Feed forward optimization                     ");
            print_progress_bar(10, (i + 1) / (float)neuron_num);
            cudaEventRecord(stop, NULL);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&et_ff, start, stop);
            et_ff /= 1000;
            printf(" | %.2f%% ET: %.2fs ETA: %.2fs", (float)(i + 1) / (float)neuron_num * 100, et_ff, et_ff / (i + 1) * neuron_num - et_ff);
            fflush(stdout);
            for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
            {
                for (unsigned long long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
                {
                    dist[graph_n_m[neuron_id][neighbour_ind]] = imax(dist[graph_n_m[neuron_id][neighbour_ind]], dist[neuron_id] + 1);
                }
            }
        }
        for (unsigned long long int j = 0; j < 400; j++)
        {
            printf("\b \b");
        }
        printf("\r");
        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&et_ff, start, stop);
        et_ff /= 1000;
        printf("\r");
        printf("%.35s", "Feed forward optimization                     ");

        printf("| %.2f%% ET: %.2fs\n", 100.0, (float)et_ff);
        fflush(stdout);

        dist_max = 0;
        for (unsigned long long int i = 0; i < neuron_num; i++)
        {
            if (dist[i] > dist_max)
            {
                dist_max = dist[i];
            }
        }

        for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            dist_extra[neuron_id] = dist[neuron_id];
        }
        // Make one extra step to check whether is a cycle in the graph
        for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                dist_extra[graph_n_m[neuron_id][neighbour_ind]] = imax(dist[graph_n_m[neuron_id][neighbour_ind]], dist[neuron_id] + 1);
            }
        }

        for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            if (dist[neuron_id] != dist_extra[neuron_id])
            {
                check_cycle = 1;
            }
        }
    }

    if ((ff_optimization > 0) && (check_cycle == 1))
    {
        program_failure("Logical error: cycle in the graph\n");
    }

    unsigned long long int *dist_number;            // count the neurons by distance
    unsigned long long int *dist_number_temp;       // temporal vector to count the neurons by distance
    unsigned long long int **dist_indices_m;        // neuron indices by distances
    unsigned long long int *dist_indices;           // the same in one vector
    unsigned long long int *first_ind_dist_indices; // pointer to the first elements in dist_indices

    // Count the neurons by distance values
    if ((ff_optimization > 0) && (check_cycle == 0))
    {
        dist_number = (unsigned long long int *)malloc((dist_max + 1) * sizeof(unsigned long long int));

        for (unsigned long long int i = 0; i <= dist_max; i++)
        {
            dist_number[i] = 0;
        }
        for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            dist_number[dist[neuron_id]]++;
        }

        dist_indices_m = allocate_imatrix(dist_max + 1, dist_number);

        // Create the list of the neuron indices by distance
        dist_number_temp = (unsigned long long int *)malloc((dist_max + 1) * sizeof(unsigned long long int));

        // Check whether is there any distance with 0 neuron
        unsigned long long int dist_min = dist_max;

        for (unsigned long long int i = 0; i < neuron_num; i++)
        {
            if (dist_number[i] < dist_min)
            {
                dist_min = dist_number[i];
            }
        }

        for (unsigned long long int i = 0; i <= dist_max; i++)
        {
            dist_number_temp[i] = 0;
        }

        for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            dist_indices_m[dist[neuron_id]][dist_number_temp[dist[neuron_id]]] = neuron_id;
            dist_number_temp[dist[neuron_id]]++;
        }

        dist_indices = (unsigned long long int *)malloc(neuron_num * sizeof(unsigned long long int)); // Ez nem jo itt!!! ???
        first_ind_dist_indices = (unsigned long long int *)malloc((dist_max + 1) * sizeof(unsigned long long int));
        for (unsigned long long int i = 0; i <= dist_max; i++)
        {
            first_ind_dist_indices[i] = 0;
        }

        // copying dist_indices_m ---> dist_indices, first_ind_dist_indices
        unsigned long long int copy_ind = 0;
        for (unsigned long long int i = 0; i <= dist_max; i++)
        {

            // Display the optimization process

            for (unsigned long long int j = 0; j < dist_number[i]; j++)
            {

                dist_indices[copy_ind] = dist_indices_m[i][j];
                copy_ind++;
            }
            if (i > 0)
            {
                first_ind_dist_indices[i] = first_ind_dist_indices[i - 1] + dist_number[i - 1];
            }
        }

        // Creating distance values for the inputs
        for (unsigned long long int i = 0; i < all_input_num; i++)
        {
            dist_input[i] = 0;
        }

        unsigned long long int counter = 0;
        for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long long int bias_id = 0; bias_id < bias_number[neuron_id]; bias_id++)
            {
                dist_input[counter] = dist[neuron_id];
                counter++;
            }
        }
    }

    //+++++++++++++++++++++++++//
    //                         //
    // Creating output headers //
    //                         //
    //+++++++++++++++++++++++++//

    unsigned long long int iter_grad = 0;

    float elapsed_time = 0.0, elapsed_time_2 = 0.0;
    FILE *f = fopen(output_name, "a");
    if (f)
    {
        time_t mytime = time(NULL);
        char *time_str = ctime(&mytime);
        time_str[strlen(time_str) - 1] = '\0';

        fprintf(f, "*************************************************************************************************************************************\n");
        fprintf(f, "|%10s |%10s |%10s |%10s |%10s |%10s |%10s |%10s |%10s |%10s |%10s | %s \n", "ITER", "MB", "LE", "VM", "BVM", "IF", "IB", "MAW", "MAB",
                "LR", "ET",
                time_str);
        fprintf(f, "-------------------------------------------------------------------------------------------------------------------------------------\n");
    }
    else
    {
        program_failure("File write error: logile\n");
    }
    fclose(f);

    // if ((strcmp(train_lossfunction_type, "bce_multilabeling") == 0) || (strcmp(train_lossfunction_type, "multilabeling_crossentropy") == 0) || (strcmp(train_lossfunction_type, "multiclassification_crossentropy") == 0))
    //{
    f = fopen(test_log, "a");
    if (f)
    {
        time_t mytime = time(NULL);
        char *time_str = ctime(&mytime);
        time_str[strlen(time_str) - 1] = '\0';
        fprintf(f, "*************************************************************************************************************************************************************\n");
        fprintf(f, "|%10s |%10s |%10s |%10s |%10s |%10s |%10s |%10s |%10s |%10s |%10s |%10s |%10s | %s \n", "ITER", "MB", "LE", "VM", "TF1", "TMCC", "BVM", "CTF1", "CTMCC", "CTACC", "CTP", "CTR", "ET", time_str);
        fprintf(f, "-------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    }
    else
    {
        program_failure("File write error: logile\n");
    }
    fclose(f);

    //}

    //++++++++++++++++++++++++++++++++++++++++++++++//
    //                                              //
    //        Copying the network to the gpu        //
    //                                              //
    //++++++++++++++++++++++++++++++++++++++++++++++//

    unsigned long long int *neighbour_number_g, *bias_number_g, *parent_number_g,
        *first_ind_neighbour_g, *first_ind_bias_g, *first_ind_parent_g,
        *activation_type_g, *graph_p_ind_n_g,
        *graph_n_g, *graph_i_g, *graph_p_g,
        *graph_logic_g, *bias_logic_g,
        *dist_indices_g, *dist_number_g, *first_ind_dist_indices_g,
        *dist_g, *dist_input_g;

    float *weight_g, *bias_g, *weight_grad_g, *bias_grad_g, *weight_temp_g, *bias_temp_g;
    float *mt_weight_g, *mt_bias_g, *vt_weight_g, *vt_bias_g;
    float *mth_weight_g, *mth_bias_g, *vth_weight_g, *vth_bias_g;
    float *ut_weight_g, *ut_bias_g;
    float *weight_trans_g;

    cudaMalloc((void **)&weight_trans_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&neighbour_number_g, sizeof(unsigned long long int) * neuron_num);
    cudaMalloc((void **)&bias_number_g, sizeof(unsigned long long int) * neuron_num);
    cudaMalloc((void **)&parent_number_g, sizeof(unsigned long long int) * all_input_num);
    cudaMalloc((void **)&first_ind_neighbour_g, sizeof(unsigned long long int) * neuron_num);
    cudaMalloc((void **)&first_ind_bias_g, sizeof(unsigned long long int) * neuron_num);
    cudaMalloc((void **)&first_ind_parent_g, sizeof(unsigned long long int) * all_input_num);
    cudaMalloc((void **)&activation_type_g, sizeof(unsigned long long int) * all_input_num);
    cudaMalloc((void **)&graph_n_g, sizeof(unsigned long long int) * all_neighbour_num);
    cudaMalloc((void **)&graph_i_g, sizeof(unsigned long long int) * all_neighbour_num);
    cudaMalloc((void **)&graph_p_g, sizeof(unsigned long long int) * all_neighbour_num);
    cudaMalloc((void **)&graph_p_ind_n_g, sizeof(unsigned long long int) * all_neighbour_num);
    cudaMalloc((void **)&graph_logic_g, sizeof(unsigned long long int) * all_neighbour_num);
    cudaMalloc((void **)&bias_logic_g, sizeof(unsigned long long int) * all_input_num);
    cudaMalloc((void **)&dist_g, sizeof(unsigned long long int) * neuron_num);
    cudaMalloc((void **)&dist_input_g, sizeof(unsigned long long int) * all_input_num);
    cudaMalloc((void **)&weight_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&bias_g, sizeof(float) * all_input_num);
    cudaMalloc((void **)&weight_temp_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&bias_temp_g, sizeof(float) * all_input_num);
    cudaMalloc((void **)&weight_grad_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&bias_grad_g, sizeof(float) * all_input_num);
    cudaMalloc((void **)&mt_weight_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&mt_bias_g, sizeof(float) * all_input_num);
    cudaMalloc((void **)&ut_weight_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&ut_bias_g, sizeof(float) * all_input_num);
    cudaMalloc((void **)&vt_weight_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&vt_bias_g, sizeof(float) * all_input_num);
    cudaMalloc((void **)&mth_weight_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&mth_bias_g, sizeof(float) * all_input_num);
    cudaMalloc((void **)&vth_weight_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&vth_bias_g, sizeof(float) * all_input_num);

    cudaMemcpy(first_ind_neighbour_g, first_ind_neighbour, sizeof(unsigned long long int) * neuron_num, cudaMemcpyHostToDevice);
    cudaMemcpy(first_ind_bias_g, first_ind_bias, sizeof(unsigned long long int) * neuron_num, cudaMemcpyHostToDevice);
    cudaMemcpy(first_ind_parent_g, first_ind_parent, sizeof(unsigned long long int) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(neighbour_number_g, neighbour_number, sizeof(unsigned long long int) * neuron_num, cudaMemcpyHostToDevice);
    cudaMemcpy(bias_number_g, bias_number, sizeof(unsigned long long int) * neuron_num, cudaMemcpyHostToDevice);
    cudaMemcpy(parent_number_g, parent_number, sizeof(unsigned long long int) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(graph_p_g, graph_p, sizeof(unsigned long long int) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(graph_p_ind_n_g, graph_p_ind_n, sizeof(unsigned long long int) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(activation_type_g, activation_type, sizeof(unsigned long long int) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(graph_n_g, graph_n, sizeof(unsigned long long int) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(graph_i_g, graph_i, sizeof(unsigned long long int) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(graph_logic_g, graph_logic, sizeof(unsigned long long int) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(bias_logic_g, bias_logic, sizeof(unsigned long long int) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(dist_g, dist, sizeof(unsigned long long int) * neuron_num, cudaMemcpyHostToDevice);
    cudaMemcpy(dist_input_g, dist_input, sizeof(unsigned long long int) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_g, weight, sizeof(float) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(bias_g, bias, sizeof(float) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_grad_g, weight_grad, sizeof(float) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(bias_grad_g, bias_grad, sizeof(float) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(mt_weight_g, mt_weight, sizeof(float) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(mt_bias_g, mt_bias, sizeof(float) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(vt_weight_g, vt_weight, sizeof(float) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(vt_bias_g, vt_bias, sizeof(float) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(mth_weight_g, mth_weight, sizeof(float) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(mth_bias_g, mth_bias, sizeof(float) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(vth_weight_g, vth_weight, sizeof(float) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(vth_bias_g, vth_bias, sizeof(float) * all_input_num, cudaMemcpyHostToDevice);

    //+++++++++++++++++++++++++//
    //                         //
    //        Main loop        //
    //                         //
    //+++++++++++++++++++++++++//

    float initial_ga = grad_alpha / div_factor;
    float initial_aa = adam_alpha / div_factor;

    float iter_forward_temp = 0.0, iter_backward_temp = 0.0;
    float iter_forward = 0.0, iter_backward = 0.0;
    iter_grad = 0;

    float grad_alpha_act = grad_alpha, adam_alpha_act = adam_alpha, adam_beta1_act = adam_beta1, adam_beta2_act = adam_beta2;
    float best_valid_metric = -inf, corr_f1score = -inf, corr_mcc = -inf, corr_accuracy = -inf, corr_precision = -inf, corr_recall = -inf;

    // Loading the training data
    FILE *f_data = fopen(input_name, "r");

    if (f_data)
    {
        float et = 0.0;
        cudaEventRecord(start, NULL);
        unsigned long long int output_num_temp = 0;

        for (unsigned long long int i = 0; i < learn_num; i++)
        {
            for (unsigned long long int j = 0; j < 400; j++)
            {
                printf("\b \b");
            }
            cudaEventRecord(stop, NULL);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&et, start, stop);
            et /= 1000;
            printf("\r");
            printf("%.35s", "Loading the training data                     ");
            print_progress_bar(10, (i + 1) / (float)learn_num);
            printf(" | %.2f%% ET: %.2fs ETA: %.2fs", (float)(i + 1) / (float)learn_num * 100, et, et / (i + 1) * learn_num - et);
            fflush(stdout);

            for (unsigned long long int j = 0; j < input_num; j++)
            {
                fscanf(f_data, "%f", &train_data[i * (input_num + output_num) + j]);
                train_data[i * (input_num + output_num) + j + input_num] = train_data[i * (input_num + output_num) + j];
            }
        }
        for (unsigned long long int j = 0; j < 400; j++)
        {
            printf("\b \b");
        }
        printf("\r");
        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&et, start, stop);
        et /= 1000;
        printf("%.35s", "Loading the training data                     ");
        printf("| %.2f%% ET: %.2fs\n", 100.0, (float)et);
        fflush(stdout);
    }
    else
    {
        program_failure("File read error in training data file!");
    }
    fclose(f_data);

    // Loading the validation data
    f_data = fopen(input_name_valid, "r");

    if (f_data)
    {
        float et = 0.0;
        cudaEventRecord(start, NULL);

        for (unsigned long long int i = 0; i < valid_num; i++)
        {
            for (unsigned long long int j = 0; j < 400; j++)
            {
                printf("\b \b");
            }
            cudaEventRecord(stop, NULL);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&et, start, stop);
            et /= 1000;
            printf("\r");
            printf("%.35s", "Loading the validation data                     ");
            print_progress_bar(10, (i + 1) / (float)valid_num);
            printf(" | %.2f%% ET: %.2fs ETA: %.2fs", (float)(i + 1) / (float)valid_num * 100, et, et / (i + 1) * test_num - et);
            fflush(stdout);

            for (unsigned long long int j = 0; j < input_num; j++)
            {
                fscanf(f_data, "%f", &valid_data[i * (input_num + output_num) + j]);
                valid_data[i * (input_num + output_num) + j + input_num] = valid_data[i * (input_num + output_num) + j];
            }
            fscanf(f_data, "%llu", &valid_labels[i]);
        }
        for (unsigned long long int j = 0; j < 400; j++)
        {
            printf("\b \b");
        }
        printf("\r");
        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&et, start, stop);
        et /= 1000;
        printf("%.35s", "Loading the validation data                     ");
        printf("| %.2f%% ET: %.2fs\n", 100.0, (float)et);
        fflush(stdout);
    }
    else
    {
        program_failure("File read error in validation data file!");
    }
    fclose(f_data);

    // Loading the validation data
    f_data = fopen(input_name_test, "r");

    if (f_data)
    {
        float et = 0.0;
        cudaEventRecord(start, NULL);

        for (unsigned long long int i = 0; i < test_num; i++)
        {
            for (unsigned long long int j = 0; j < 400; j++)
            {
                printf("\b \b");
            }
            cudaEventRecord(stop, NULL);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&et, start, stop);
            et /= 1000;
            printf("\r");
            printf("%.35s", "Loading the test data                     ");
            print_progress_bar(10, (i + 1) / (float)test_num);
            printf(" | %.2f%% ET: %.2fs ETA: %.2fs", (float)(i + 1) / (float)test_num * 100, et, et / (i + 1) * test_num - et);
            fflush(stdout);

            for (unsigned long long int j = 0; j < input_num; j++)
            {
                fscanf(f_data, "%f", &test_data[i * (input_num + output_num) + j]);
                test_data[i * (input_num + output_num) + j + input_num] = test_data[i * (input_num + output_num) + j];
            }
            fscanf(f_data, "%llu", &test_labels[i]);
        }
        for (unsigned long long int j = 0; j < 400; j++)
        {
            printf("\b \b");
        }
        printf("\r");
        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&et, start, stop);
        et /= 1000;
        printf("%.35s", "Loading the test data                    ");
        printf("| %.2f%% ET: %.2fs\n", 100.0, (float)et);
        fflush(stdout);
    }
    else
    {
        program_failure("File read error in test data file!");
    }

    fclose(f_data);

    float lr;

    // valid_th_best = -inf;
    unsigned long long int early_stopping_counter = 0;

    /*
    unsigned long long int res = 0;
    for (unsigned long long int i=0;i<valid_num;i++){
        res += valid_labels[i];
    }


    printf(" %llu - %llu ", res,valid_num );
    res = 0;
    for (unsigned long long int i=0;i<valid_num;i++){
        res += test_labels[i];
    }
    printf(" %llu - %llu ", res,test_num );
    exit(0);
    */

    float iter_forward_s;
    float iter_backward_s;
    float error_learn;

    while (iter_grad < maxiter_grad)
    {
        iter_grad++;

        cudaEventRecord(start, NULL);

        float elapsed_time_temp = 0.0;

        iter_backward = 0.0;
        iter_forward = 0.0;
        float error_temp_mean = 0.0;
        char str_valid[2000] = "";
        unsigned long long int train_data_num = 0;
        float max_w, max_b;

        //+++++++++++++++++++++++++//
        //                         //
        //         Shuffling       //
        //                         //
        //+++++++++++++++++++++++++//

        for (unsigned long long int shuffle_id = 0; shuffle_id < shuffle_num; shuffle_id++)
        {
            unsigned long long int ind_a = rand_range_int(0, learn_num - 1);
            unsigned long long int ind_b = rand_range_int(0, learn_num - 1);
            for (unsigned long long int col_ind = 0; col_ind < input_num + output_num; col_ind++)
            {
                float val_temp = train_data[ind_a * (input_num + output_num) + col_ind];
                train_data[ind_a * (input_num + output_num) + col_ind] = train_data[ind_b * (input_num + output_num) + col_ind];
                train_data[ind_b * (input_num + output_num) + col_ind] = val_temp;
            }
        }

        float grad_alpha_min = grad_alpha / final_div_factor;
        float adam_alpha_min = adam_alpha / final_div_factor;

        //+++++++++++++++++++++++++//
        //                         //
        //         Training        //
        //                         //
        //+++++++++++++++++++++++++//

        for (unsigned long long int mini_batch_id = 0; mini_batch_id < mini_batch_num; mini_batch_id++)
        {
            // Load a mini-batch
            unsigned long long int mini_batch_len;
            unsigned long long int mini_batch_si = mini_batch_id * mini_batch_size;
            unsigned long long int mini_batch_ei = (mini_batch_id + 1) * mini_batch_size - 1;
            if (mini_batch_ei > learn_num - 1)
            {
                mini_batch_ei = learn_num - 1;
            }
            mini_batch_len = mini_batch_ei - mini_batch_si + 1;
            train_data_num += mini_batch_len;
            for (unsigned long long int row_ind = mini_batch_si; row_ind <= mini_batch_ei; row_ind++)
            {
                for (unsigned long long int col_ind = 0; col_ind < input_num + output_num; col_ind++)
                {
                    datas_mini_batch[(row_ind - mini_batch_si) * (input_num + output_num) + col_ind] = train_data[row_ind * (input_num + output_num) + col_ind];
                }
            }

            // Learning rate scheduler
            unsigned long long int act_step;
            unsigned long long int all_step;
            unsigned long long int cut;
            switch (lr_scheduler)
            {
            case 0:
                grad_alpha_act = grad_alpha; // * pow(lr_gamma, power);
                adam_alpha_act = adam_alpha; // * pow(lr_gamma, power);
                break;
            case 1:
                act_step = mini_batch_id + 1 + (iter_grad - 1) * mini_batch_num;
                all_step = maxiter_grad * mini_batch_num;

                cut = all_step * pcr;
                // 0.3 * all_step --- avoiding the use of float numbers here
                // printf("\n cut %f all_step %llu all_step*cut %f \n",cut,all_step,all_step*cut);
                if (act_step < cut)
                {
                    grad_alpha_act = f_lr_momentum(act_step, grad_alpha / div_factor, grad_alpha, cut);
                    adam_alpha_act = f_lr_momentum(act_step, adam_alpha / div_factor, adam_alpha, cut);
                }
                else
                {
                    grad_alpha_act = f_lr_momentum(act_step - cut, grad_alpha, initial_ga / final_div_factor, all_step - cut);
                    adam_alpha_act = f_lr_momentum(act_step - cut, adam_alpha, initial_aa / final_div_factor, all_step - cut);
                }

                if (cyclic_momentum)
                {
                    if (act_step < cut)
                    {
                        adam_beta1_act = f_lr_momentum(act_step, max_momentum, base_momentum, cut);
                    }
                    else
                    {
                        adam_beta1_act = f_lr_momentum(act_step - cut, base_momentum, max_momentum, all_step - cut);
                    }
                }

                break;
            case 2:
                // https://paperswithcode.com/method/cosine-annealing
                act_step = mini_batch_id + 1 + (iter_grad - 1) * mini_batch_num;
                all_step = maxiter_grad * mini_batch_num;
                cut = all_step * pcr + 1;
                act_step = act_step % cut;

                grad_alpha_act = grad_alpha_min + 0.5 * (grad_alpha - grad_alpha_min) * (1.0 + cosf((float)(act_step) / (float)(cut)*M_PI));
                adam_alpha_act = adam_alpha_min + 0.5 * (adam_alpha - adam_alpha_min) * (1.0 + cosf((float)(act_step) / (float)(cut)*M_PI));

                if (cyclic_momentum)
                {
                    // if (act_step < cut)
                    //{
                    //     adam_beta1_act = f_lr_momentum(act_step, max_momentum, base_momentum, cut);
                    // }
                    // else
                    //{
                    adam_beta1_act = f_lr_momentum(act_step - cut, base_momentum, max_momentum, all_step - cut);
                    //}
                }

                // grad_alpha_act = f_lr_momentum(act_step, grad_alpha, grad_alpha / final_div_factor, cut);
                // adam_alpha_act = f_lr_momentum(act_step, adam_alpha, adam_alpha / final_div_factor, cut);
                // grad_alpha_act = f_lr_momentum(act_step, grad_alpha / div_factor, grad_alpha, cut);
                // adam_alpha_act = f_lr_momentum(act_step, adam_alpha / div_factor, adam_alpha, cut);
                //  printf("\n all_step*cut=cut %llu all_step %llu \n",cut,all_step);
                //  printf("g %f a %f\n",grad_alpha_act, adam_alpha_act);
                break;
            case 3:
                unsigned long long int power = 0;
                // if (iter_grad  > 1)
                //{
                power = (iter_grad - 1) / step_size;
                //}
                grad_alpha_act = grad_alpha * pow(lr_gamma, power);
                adam_alpha_act = adam_alpha * pow(lr_gamma, power);

                break;
            }

            if (optimizer == 1)
            {
                lr = grad_alpha_act;
            }
            if ((optimizer == 2) || (optimizer == 3))
            {
                lr = adam_alpha_act;
            }

            // Nesterov
            if ((nesterov == 1) && (optimizer == 1))
            {
                cudaMemcpy(weight_temp_g, weight_g, sizeof(float) * all_neighbour_num, cudaMemcpyDeviceToDevice);
                cudaMemcpy(bias_temp_g, bias_g, sizeof(float) * all_input_num, cudaMemcpyDeviceToDevice);
                update_weight_gd_h_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_temp_g, bias_temp_g, weight_grad_g, bias_grad_g, vt_weight_g, vt_bias_g, graph_logic_g, bias_logic_g, all_input_num, all_neighbour_num, grad_alpha, adam_beta1);
                update_bias_gd_h_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(weight_temp_g, bias_temp_g, weight_grad_g, bias_grad_g, vt_weight_g, vt_bias_g, graph_logic_g, bias_logic_g, all_input_num, all_neighbour_num, grad_alpha, adam_beta1);
            }
            else
            {
                cudaMemcpy(weight_temp_g, weight_g, sizeof(float) * all_neighbour_num, cudaMemcpyDeviceToDevice);
                cudaMemcpy(bias_temp_g, bias_g, sizeof(float) * all_input_num, cudaMemcpyDeviceToDevice);
            }

            // Calculating the gradient on the mini-batch

            float error_temp = 0.0;
            if (ff_optimization == 0)
            {
                error_temp = calc_gradient_mini_batch(datas_mini_batch, mini_batch_len,
                                                      neighbour_number_g, neighbour_number, bias_number_g, bias_number, parent_number_g,
                                                      first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias, first_ind_parent_g,
                                                      activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
                                                      graph_p_g, graph_p_ind_n_g, weight_temp_g, bias_temp_g,
                                                      weight_grad_g, bias_grad_g, &iter_forward_temp, &iter_backward_temp);
            }
            else
            {
                error_temp = calc_gradient_mini_batch_ff(datas_mini_batch, mini_batch_len,
                                                         neighbour_number_g, neighbour_number, bias_number_g, bias_number, parent_number_g,
                                                         first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias, first_ind_parent_g,
                                                         activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
                                                         graph_p_g, graph_p_ind_n_g, weight_temp_g, bias_temp_g,
                                                         weight_grad_g, bias_grad_g, &iter_forward_temp, &iter_backward_temp,
                                                         dist_max, dist_g, dist, dist_input_g, dist_input);
            }
            error_temp *= mini_batch_len;
            error_temp_mean += error_temp;
            iter_forward += iter_forward_temp;
            iter_backward += iter_backward_temp;

            // Copying from gpu
            cudaMemcpy(weight_grad, weight_grad_g, sizeof(float) * all_neighbour_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(bias_grad, bias_grad_g, sizeof(float) * all_input_num, cudaMemcpyDeviceToHost);

            cudaThreadSynchronize();

            // Max weight gradient
            max_w = calc_vector_norm(weight_grad, all_neighbour_num);

            // Max bias gradient value
            max_b = calc_vector_norm(bias_grad, all_input_num);

            // Shared weights and biases
            if (shared_weights_num > 0)
            {

                for (unsigned long long int group_id = 0; group_id < shared_weights_num; group_id++)
                {
                    // Collect the shared weight gradients
                    shared_weights_values[group_id] = 0.0;
                    for (unsigned long long int weight_id = 0; weight_id < shared_weights_blocksize[group_id]; weight_id++)
                    {
                        unsigned long long int startind = shared_weights[group_id][weight_id * 3];
                        startind = first_ind_neighbour[startind];
                        unsigned long long int endind = shared_weights_indices[group_id][weight_id];

                        shared_weights_values[group_id] += weight_grad[startind + endind];
                        // shared_weights_values[group_id] += weight_grad[shared_weights[group_id][weight_id * 3]][shared_weights_indices[group_id][weight_id]];
                    }

                    // Copying back the collected shared weight gradients
                    for (unsigned long long int weight_id = 0; weight_id < shared_weights_blocksize[group_id]; weight_id++)
                    {
                        unsigned long long int startind = shared_weights[group_id][weight_id * 3];
                        startind = first_ind_neighbour[startind];
                        unsigned long long int endind = shared_weights_indices[group_id][weight_id];

                        weight_grad[startind + endind] = shared_weights_values[group_id];
                    }
                }

                // Copying back to gpu
                cudaMemcpy(weight_grad_g, weight_grad, sizeof(float) * all_neighbour_num, cudaMemcpyHostToDevice);
            }

            if (shared_biases_num > 0)
            {

                for (unsigned long long int group_id = 0; group_id < shared_biases_num; group_id++)
                {
                    // Collect the shared bias gradients
                    shared_biases_values[group_id] = 0.0;
                    for (unsigned long long int bias_id = 0; bias_id < shared_biases_blocksize[group_id]; bias_id++)
                    {
                        unsigned long long int startind = shared_biases[group_id][bias_id * 2];
                        startind = first_ind_bias[startind];
                        unsigned long long int endind = shared_biases[group_id][bias_id * 2 + 1];

                        shared_biases_values[group_id] += bias_grad[startind + endind];
                    }

                    // Copying back the collected shared bias gradients
                    for (unsigned long long int bias_id = 0; bias_id < shared_biases_blocksize[group_id]; bias_id++)
                    {
                        unsigned long long int startind = shared_biases[group_id][bias_id * 2];
                        startind = first_ind_bias[startind];
                        unsigned long long int endind = shared_biases[group_id][bias_id * 2 + 1];

                        bias_grad[startind + endind] = shared_biases_values[group_id];
                    }
                }

                // Copying back to gpu
                cudaMemcpy(bias_grad_g, bias_grad, sizeof(float) * all_input_num, cudaMemcpyHostToDevice);
            }

            cudaEventRecord(stop, NULL);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time_temp, start, stop);
            elapsed_time_temp /= 1000;

            // Update the weights
            switch (optimizer)
            {
            case 1:
                update_weight_gd_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_g, bias_g, weight_grad_g, bias_grad_g, vt_weight_g, vt_bias_g, graph_logic_g, bias_logic_g, all_input_num, all_neighbour_num, grad_alpha_act, adam_beta1_act, nesterov);
                update_bias_gd_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(weight_g, bias_g, weight_grad_g, bias_grad_g, vt_weight_g, vt_bias_g, graph_logic_g, bias_logic_g, all_input_num, all_neighbour_num, grad_alpha_act, adam_beta1_act, nesterov);
                break;
            case 2:
                adam_beta1t *= adam_beta1_act;
                adam_beta2t *= adam_beta2_act;
                update_weight_adam_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_g, bias_g, weight_grad_g, bias_grad_g, mt_weight_g, mt_bias_g, vt_weight_g, vt_bias_g, mth_weight_g, mth_bias_g, vth_weight_g, vth_bias_g, graph_logic_g, bias_logic_g, all_input_num, all_neighbour_num, adam_alpha_act, adam_beta1_act, adam_beta2_act, adam_beta1t, adam_beta2t, adam_eps);
                update_bias_adam_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(weight_g, bias_g, weight_grad_g, bias_grad_g, mt_weight_g, mt_bias_g, vt_weight_g, vt_bias_g, mth_weight_g, mth_bias_g, vth_weight_g, vth_bias_g, graph_logic_g, bias_logic_g, all_input_num, all_neighbour_num, adam_alpha_act, adam_beta1_act, adam_beta2_act, adam_beta1t, adam_beta2t, adam_eps);
                break;
            case 3:
                adam_beta1t *= adam_beta1_act;
                adam_beta2t *= adam_beta2_act;
                update_weight_adamax_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_g, bias_g, weight_grad_g, bias_grad_g, mt_weight_g, mt_bias_g, ut_weight_g, ut_bias_g, graph_logic_g, bias_logic_g, all_input_num, all_neighbour_num, adam_alpha_act, adam_beta1_act, adam_beta2_act, adam_beta1t, adam_beta2t, adam_eps);
                update_bias_adamax_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(weight_g, bias_g, weight_grad_g, bias_grad_g, mt_weight_g, mt_bias_g, ut_weight_g, ut_bias_g, graph_logic_g, bias_logic_g, all_input_num, all_neighbour_num, adam_alpha_act, adam_beta1_act, adam_beta2_act, adam_beta1t, adam_beta2t, adam_eps);
                break;
            }

            iter_forward_s = iter_forward / (mini_batch_id + 1);
            iter_backward_s = iter_backward / (mini_batch_id + 1);
            error_learn = error_temp_mean / (train_data_num * output_num);

            // Display the training results
            for (unsigned long long int i = 0; i < 400; i++)
            {
                printf("\b \b");
            }
            printf("\r");

            if (mini_batch_id < mini_batch_num - 1)
            {
                print_progress_bar(10, (mini_batch_id + 1) / (float)mini_batch_num);
                printf(" [%llu/%llu] LE: %.2E ET: %.1fs ETA: %.1fs | ", mini_batch_id + 1,
                       mini_batch_num, error_temp / (mini_batch_len * output_num),
                       elapsed_time_temp, elapsed_time_temp * mini_batch_num / (mini_batch_id + 1) - elapsed_time_temp + 0.01);
                print_progress_bar(10, iter_grad / (float)maxiter_grad);
                printf("%s", str_valid);
            }
            else
            {
                print_progress_bar(10, (mini_batch_id + 1) / (float)mini_batch_num);
                printf(" [%llu/%llu] LE: %.2E ET: %.1fs | ", mini_batch_id + 1,
                       mini_batch_num, error_learn,
                       elapsed_time_temp);
                print_progress_bar(10, iter_grad / (float)maxiter_grad);
                printf("%s", str_valid);
            }
            fflush(stdout);
            // char str_valid[1000] = "";

            //+++++++++++++++++++++++++//
            //                         //
            //  Validation - logging   //
            //                         //
            //+++++++++++++++++++++++++//

            float error_valid = 0.0, error_test = 0.0;
            float acc_learn = 0.0, acc_valid = 0.0, acc_test = 0.0;

            if (((mini_batch_id + 1) % sfreq == 0) || (mini_batch_id == mini_batch_num - 1))
            {
                // Transposing the weight matrix
                weight_transpose_gpu<<<(neuron_num + TPB - 1) / TPB, TPB>>>(first_ind_neighbour_g, first_ind_bias_g, first_ind_parent_g,
                                                                            neighbour_number_g, bias_number_g, parent_number_g, graph_p_g, graph_p_ind_n_g, weight_g, weight_trans_g, neuron_num);

                predict(train_data, valid_data, test_data,
                        mini_batch_num, mini_batch_num_valid, mini_batch_num_test, neighbour_number_g, neighbour_number, bias_number_g, bias_number,
                        parent_number_g, first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias,
                        first_ind_parent_g, activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
                        graph_p_g, graph_p_ind_n_g, weight_g, weight_trans_g, bias_g, dist_max, dist_g, dist, dist_input_g,
                        dist_input, predictions_valid, predictions_test, errors_valid, errors_test);

                // Calculating the Z-score of the errors on the normal valid data (valid_labels[i]=0)
                valid_std = 0.0;
                valid_mean = 0.0;
                unsigned long long int valid_std_n = 0;

                for (unsigned long long int v_id = 0; v_id < valid_num; v_id++)
                {
                    if (valid_labels[v_id] == 0)
                    {
                        valid_std_n += 1;
                        valid_mean += errors_valid[v_id];
                    }
                }
                valid_mean /= valid_std_n;

                for (unsigned long long int v_id = 0; v_id < valid_num; v_id++)
                {
                    if (valid_labels[v_id] == 0)
                    {

                        valid_std += powf(errors_valid[v_id] - valid_mean, 2.0);
                    }
                }
                valid_std /= valid_std_n - 1;
                valid_std = sqrtf(valid_std);
                // printf("\n valid_mean %.5f valid_std: %.5f \n", valid_mean,valid_std);

                // Normalizing the errors_valid vector with these values
                for (unsigned long long int v_id = 0; v_id < valid_num; v_id++)
                {
                    errors_valid[v_id] = (errors_valid[v_id] - valid_mean) / valid_std;
                }

                // For loop in [-4,4] by 0.01 step size
                //      and calculating the predictions_valid_temp

                // printf("\n");
                unsigned long long int nthreads;
#pragma omp parallel
                {

                    unsigned long long int id = omp_get_thread_num();

                    if (id == 0)
                    {
                        nthreads = omp_get_num_threads();
                    }

                    unsigned long long int t_id;
                    unsigned long long int *valid_pred_labels = (unsigned long long int *)malloc(valid_num * sizeof(unsigned long long int));

#pragma omp barrier
                    for (t_id = id; t_id < (range_div + 1); t_id = t_id + nthreads)
                    {
                        float th = -4.0 + t_id * 8.0 / (float)(range_div);

                        for (unsigned long long int v_id = 0; v_id < valid_num; v_id++)
                        {
                            if (errors_valid[v_id] > th)
                            {
                                valid_pred_labels[v_id] = 1;
                            }
                            else
                            {
                                valid_pred_labels[v_id] = 0;
                            }
                        }
                        float tpr, fpr, f1score, accuracy, precision;
                        // TPR, FPR, F1-score
                        calculate_tpr_fpr_bc(valid_pred_labels, valid_labels, valid_num, &tpr, &fpr, &f1score, &accuracy, &precision);
                        valid_fpr[t_id] = fpr;
                        valid_tpr[t_id] = tpr;

                        // MCC
                        float mcc_th = calculate_mcc(valid_pred_labels, valid_labels, valid_num);

                        mcc_list[t_id] = mcc_th;
                        f1score_list[t_id] = f1score;
                    }
                    free(valid_pred_labels);
                }

                float valid_th_f1 = -inf, valid_th_mcc = -inf, valid_th_tpr = -inf;
                float best_f1_metric = -inf, best_mcc_metric = -inf, best_tpr_metric = -inf;

                for (unsigned long long int t_id = 0; t_id < (range_div + 1); t_id++)
                {
                    float th = -4.0 + t_id * 8.0 / (float)(range_div);
                    float mcc_th = mcc_list[t_id];
                    float f1score = f1score_list[t_id];
                    float tpr_temp = valid_tpr[t_id];

                    if (f1score > best_f1_metric)
                    {
                        best_f1_metric = f1score;
                        valid_th_f1 = th;
                    }
                    if (mcc_th > best_mcc_metric)
                    {
                        best_mcc_metric = mcc_th;
                        valid_th_mcc = th;
                    }
                    if (tpr_temp > best_tpr_metric)
                    {
                        best_tpr_metric = tpr_temp;
                        valid_th_tpr = th;
                    }
                }
                if (strcmp(valid_metric_type_2, "MCC") == 0)
                {
                    valid_th = valid_th_mcc;
                }
                if (strcmp(valid_metric_type_2, "F1") == 0)
                {
                    valid_th = valid_th_f1;
                }
                if (strcmp(valid_metric_type_2, "TPR") == 0)
                {
                    valid_th = valid_th_tpr;
                }

                if (strcmp(valid_metric_type, "AUC") == 0)
                {
                    valid_metric = calculate_auc(valid_fpr, valid_tpr, (range_div + 1));
                }
                if (strcmp(valid_metric_type, "MCC") == 0)
                {
                    valid_metric = best_mcc_metric;
                }
                if (strcmp(valid_metric_type, "F1") == 0)
                {
                    valid_metric = best_f1_metric;
                }
                if (strcmp(valid_metric_type, "TPR") == 0)
                {
                    valid_metric = best_tpr_metric;
                }

                // Normalizing the errors_valid vector with these values
                for (unsigned long long int v_id = 0; v_id < test_num; v_id++)
                {
                    errors_test[v_id] = (errors_test[v_id] - valid_mean) / valid_std;
                }

                for (unsigned long long int v_id = 0; v_id < test_num; v_id++)
                {
                    if (errors_test[v_id] > valid_th)
                    {
                        test_pred_labels[v_id] = 1;
                    }
                    else
                    {
                        test_pred_labels[v_id] = 0;
                    }
                }
                float tpr, fpr, f1score, accuracy, precision;
                calculate_tpr_fpr_bc(test_pred_labels, test_labels, test_num, &tpr, &fpr, &f1score, &accuracy, &precision);

                // mcc-t szamolunk
                float mcc_th = calculate_mcc(test_pred_labels, test_labels, test_num);

                //+++++++++++++++++++++++++//
                //                         //
                //        Best Model       //
                //                         //
                //+++++++++++++++++++++++++//

                if (valid_metric > best_valid_metric)
                {
                    // printf("\n mcc: %10.6f tpr: %10.6f vm: %10.6f bvm: %10.6f \n", mcc_th, tpr, valid_metric, best_valid_metric);
                    early_stopping_counter = 0;

                    best_valid_metric = valid_metric;
                    valid_th_best = valid_th;
                    valid_mean_best = valid_mean;
                    valid_std_best = valid_std;

                    corr_f1score = f1score;
                    corr_mcc = mcc_th;
                    corr_accuracy = accuracy;
                    corr_precision = precision;
                    corr_recall = tpr;

                    cudaMemcpy(weight_best, weight_g, sizeof(float) * all_neighbour_num, cudaMemcpyDeviceToHost);
                    cudaMemcpy(bias_best, bias_g, sizeof(float) * all_input_num, cudaMemcpyDeviceToHost);
                    cudaThreadSynchronize();
                }
                else
                {
                    early_stopping_counter++;
                }
                if ((early_stopping > 0) && (early_stopping_counter == early_stopping))
                {
                    printf("\nEarly Stopping limit reached.\n");
                    save_weight_bias(save_best_model, weight_best, bias_best, neuron_num, neighbour_number, bias_number,
                                     first_ind_neighbour, first_ind_bias);
                    break;
                }

                cudaEventRecord(stop, NULL);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&elapsed_time_2, start, stop);
                elapsed_time_2 /= 1000;

                // Display the results
                // printf("\r");
                for (unsigned long long int i = 0; i < 800; i++)
                {
                    printf("\b \b");
                }
                printf("\r");
                if (mini_batch_id < mini_batch_num - 1)
                {
                    print_progress_bar(10, (mini_batch_id + 1) / (float)mini_batch_num);
                    printf(" [%llu/%llu] LE: %.2E ET: %.1fs ETA: %.1fs", mini_batch_id + 1,
                           mini_batch_num, error_temp / mini_batch_len,
                           elapsed_time_temp, elapsed_time_temp * mini_batch_num / (mini_batch_id + 1) - elapsed_time_temp + 0.01);
                }
                else
                {
                    print_progress_bar(10, (mini_batch_id + 1) / (float)mini_batch_num);
                    printf(" [%llu/%llu] LE: %.2E ET: %.1fs", mini_batch_id + 1,
                           mini_batch_num, error_learn,
                           elapsed_time_temp);
                }

                printf(" | ");
                print_progress_bar(10, iter_grad / (float)maxiter_grad);

                sprintf(str_valid, " %3llu% [%llu/%llu] LE: %.2E VM: %.3f MCC: %.3f TF1: %.3f | BVM: %.3f | ET: %.1fs", iter_grad * 100 / maxiter_grad, iter_grad,
                        maxiter_grad, error_learn, valid_metric, mcc_th, f1score, best_valid_metric, elapsed_time + elapsed_time_temp);
                printf("%s", str_valid);

                fflush(stdout);

                f = fopen(output_name, "a");
                if (f)
                {
                    fprintf(f, "|%10u |%10u |%10.2E |%10.6f |%10.6f |%10.2f |%10.2f |%10.6f |%10.6f |%10.2E |%10.2f | \n",
                            iter_grad, mini_batch_id + 1, error_learn, valid_metric, best_valid_metric, iter_forward_s, iter_backward_s, max_w, max_b, lr, elapsed_time + elapsed_time_temp);
                }
                else
                {
                    program_failure("File write error: logfile\n");
                }
                fclose(f);

                if ((fabsf(error_learn) > inf / 2.0) || (isnan(error_learn)))
                {
                    printf("\nExploding iteration.\n");
                    save_weight_bias(save_best_model, weight_best, bias_best, neuron_num, neighbour_number, bias_number,
                                     first_ind_neighbour, first_ind_bias);
                    break;
                }

                //+++++++++++++++++++++++++//
                //                         //
                //     Saving to file      //
                //                         //
                //+++++++++++++++++++++++++//

                FILE *f_test = fopen(test_log, "a");
                if (f_test)
                {
                    fprintf(f_test, "|%10d |%10d |%10.6f |%10.6f |%10.6f |%10.6f |%10.6f |%10.6f |%10.6f |%10.6f |%10.6f |%10.6f |%10.2f | \n",
                            iter_grad, mini_batch_id + 1, error_learn, valid_metric, f1score, mcc_th, best_valid_metric, corr_f1score, corr_mcc, corr_accuracy, corr_precision, corr_recall, elapsed_time + elapsed_time_temp);
                    //        "ITER",        "MB",            "LE",        "VM",       "TF1", "TMCC",  "ET", time_str
                    // iter_grad, maxiter_grad, error_learn, valid_metric, test_metric, best_valid_metric, corr_f1score, elapsed_time + elapsed_time_temp
                }
                else
                {
                    program_failure("File write error: logile\n");
                }
                fclose(f_test);
            }
        }
        if (((early_stopping > 0) && (early_stopping_counter == early_stopping)) || (fabsf(error_learn) > inf / 2.0) || (isnan(error_learn)))
        {
            break;
        }
        elapsed_time += elapsed_time_temp;
        printf("\n");

        //+++++++++++++++++++++++++//
        //                         //
        //          Saving         //
        //                         //
        //+++++++++++++++++++++++++//

        save_weight_bias(save_best_model, weight_best, bias_best, neuron_num, neighbour_number, bias_number,
                         first_ind_neighbour, first_ind_bias);
    }
    cudaEventRecord(start, NULL);

    //+++++++++++++++++++++++++//
    //                         //
    //       Predictions       //
    //                         //
    //+++++++++++++++++++++++++//

    // weight_best <--- weights
    // bias_best <--- bias

    if (loaddatas == 1)
    {
        cudaMemcpy(weight_g, weight, sizeof(float) * all_neighbour_num, cudaMemcpyHostToDevice);
        cudaMemcpy(bias_g, bias, sizeof(float) * all_input_num, cudaMemcpyHostToDevice);

        save_weight_bias(save_best_model, weight, bias, neuron_num, neighbour_number, bias_number,
                         first_ind_neighbour, first_ind_bias);
    }
    else
    {
        cudaMemcpy(weight_g, weight_best, sizeof(float) * all_neighbour_num, cudaMemcpyHostToDevice);
        cudaMemcpy(bias_g, bias_best, sizeof(float) * all_input_num, cudaMemcpyHostToDevice);
        save_weight_bias(save_best_model, weight_best, bias_best, neuron_num, neighbour_number, bias_number,
                         first_ind_neighbour, first_ind_bias);
    }

    // Copying back to gpu

    weight_transpose_gpu<<<(neuron_num + TPB - 1) / TPB, TPB>>>(first_ind_neighbour_g, first_ind_bias_g, first_ind_parent_g,
                                                                neighbour_number_g, bias_number_g, parent_number_g, graph_p_g, graph_p_ind_n_g, weight_g, weight_trans_g, neuron_num);

    predict(train_data, valid_data, test_data,
            mini_batch_num, mini_batch_num_valid, mini_batch_num_test, neighbour_number_g, neighbour_number, bias_number_g, bias_number,
            parent_number_g, first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias,
            first_ind_parent_g, activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
            graph_p_g, graph_p_ind_n_g, weight_g, weight_trans_g, bias_g, dist_max, dist_g, dist, dist_input_g,
            dist_input, predictions_valid, predictions_test, errors_valid, errors_test);

    // Saving the predictions without scaling for further work in python

    FILE *f_predict = fopen(predict_name_valid, "w");
    if (f_predict)
    {
        for (unsigned long long int i = 0; i < valid_num; i++)
        {
            for (unsigned long long int j = 0; j < output_num; j++)
            {
                fprintf(f_predict, "%f ", predictions_valid[i][j]);
            }
            fprintf(f_predict, "\n");
        }
    }
    fclose(f_predict);

    f_predict = fopen(predict_name_test, "w");
    if (f_predict)
    {
        for (unsigned long long int i = 0; i < test_num; i++)
        {
            for (unsigned long long int j = 0; j < output_num; j++)
            {
                fprintf(f_predict, "%f ", predictions_test[i][j]);
            }
            fprintf(f_predict, "\n");
        }
    }
    fclose(f_predict);

    // Normalizing the errors_valid vector
    for (unsigned long long int v_id = 0; v_id < valid_num; v_id++)
    {
        errors_valid[v_id] = (errors_valid[v_id] - valid_mean_best) / valid_std_best;
    }

    unsigned long long int nthreads;
#pragma omp parallel
    {

        unsigned long long int id = omp_get_thread_num();

        if (id == 0)
        {
            nthreads = omp_get_num_threads();
        }

        unsigned long long int t_id;
        unsigned long long int *valid_pred_labels = (unsigned long long int *)malloc(valid_num * sizeof(unsigned long long int));

#pragma omp barrier
        for (t_id = id; t_id < (range_div + 1); t_id = t_id + nthreads)
        {
            float th = -4.0 + t_id * 8.0 / (float)(range_div);

            for (unsigned long long int v_id = 0; v_id < valid_num; v_id++)
            {
                if (errors_valid[v_id] > th)
                {
                    valid_pred_labels[v_id] = 1;
                }
                else
                {
                    valid_pred_labels[v_id] = 0;
                }
            }
            float tpr, fpr, f1score, accuracy, precision;
            // TPR, FPR, F1-score
            calculate_tpr_fpr_bc(valid_pred_labels, valid_labels, valid_num, &tpr, &fpr, &f1score, &accuracy, &precision);
            valid_fpr[t_id] = fpr;
            valid_tpr[t_id] = tpr;

            // MCC
            float mcc_th = calculate_mcc(valid_pred_labels, valid_labels, valid_num);

            mcc_list[t_id] = mcc_th;
            f1score_list[t_id] = f1score;
        }
        free(valid_pred_labels);
    }

    float v_auc = calculate_auc(valid_fpr, valid_tpr, (range_div + 1));

    // Normalizing the errors_test vector
    for (unsigned long long int v_id = 0; v_id < test_num; v_id++)
    {
        errors_test[v_id] = (errors_test[v_id] - valid_mean_best) / valid_std_best;
    }

    unsigned long long int *valid_pred_labels = (unsigned long long int *)malloc(valid_num * sizeof(unsigned long long int));
    for (unsigned long long int v_id = 0; v_id < valid_num; v_id++)
    {
        if (errors_valid[v_id] > valid_th_best)
        {
            valid_pred_labels[v_id] = 1;
        }
        else
        {
            valid_pred_labels[v_id] = 0;
        }
    }

    for (unsigned long long int v_id = 0; v_id < test_num; v_id++)
    {
        if (errors_test[v_id] > valid_th_best)
        {
            test_pred_labels[v_id] = 1;
        }
        else
        {
            test_pred_labels[v_id] = 0;
        }
    }

    // Final Test metrics calculations
    float tpr, fpr, f1score, v_tpr, v_fpr, v_f1score, v_mcc, t_f1score, t_mcc, t_tpr, t_fpr;
    float v_accuracy, v_precision, t_accuracy, t_precision;

    calculate_tpr_fpr_bc(valid_pred_labels, valid_labels, valid_num, &v_tpr, &v_fpr, &v_f1score, &v_accuracy, &v_precision);
    calculate_tpr_fpr_bc(test_pred_labels, test_labels, test_num, &t_tpr, &t_fpr, &t_f1score, &t_accuracy, &t_precision);

    // MCC
    v_mcc = calculate_mcc(valid_pred_labels, valid_labels, valid_num);
    t_mcc = calculate_mcc(test_pred_labels, test_labels, test_num);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_2, start, stop);
    elapsed_time_2 /= 1000;

    f = fopen(test_log_final, "a");

    if (f)
    {
        time_t mytime = time(NULL);
        char *time_str = ctime(&mytime);
        time_str[strlen(time_str) - 1] = '\0';
        fprintf(f, "*************************************************************************************************************************\n");
        fprintf(f, "|%10s |%10s |%10s |%10s |%10s |%10s |%10s |%10s |%10s |%10s | %s \n", "ITER", "VAUC", "VMCC", "VF1", "CTF1", "CTMCC", "CTACC", "CTP", "CTR", "ET", time_str);
        fprintf(f, "-------------------------------------------------------------------------------------------------------------------------\n");
        fprintf(f, "|%10d |%10.6f |%10.6f |%10.6f |%10.6f |%10.6f |%10.6f |%10.6f |%10.6f |%10.2f | \n",
                iter_grad, v_auc, v_mcc, v_f1score, t_f1score, t_mcc, t_accuracy, t_precision, t_tpr, elapsed_time + elapsed_time_2);
        fprintf(f, "-------------------------------------------------------------------------------------------------------------------------\n");
    }
    else
    {
        program_failure("File write error: logile\n");
    }
    fclose(f);

    // printf("\n FINAL | F1: %.5f MCC: %.5f\n", f1score, mcc_th);

    f = fopen(output_name, "a");
    if (f)
    {
        fprintf(f, "-------------------------------------------------------------------------------------------------------------------------------------\n");
    }
    else
    {
        program_failure("File write error: logile\n");
    }
    fclose(f);
    // if ((strcmp(train_lossfunction_type, "bce_multilabeling") == 0) || (strcmp(train_lossfunction_type, "multilabeling_crossentropy") == 0) || (strcmp(train_lossfunction_type, "multiclassification_crossentropy") == 0))
    //{
    f = fopen(test_log, "a");
    if (f)
    {
        fprintf(f, "------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    }
    else
    {
        program_failure("File write error: logile\n");
    }
    //}

    cudaFree(weight_trans_g);
    cudaFree(neighbour_number_g);
    cudaFree(bias_number_g);
    cudaFree(parent_number_g);
    cudaFree(first_ind_neighbour_g);
    cudaFree(first_ind_bias_g);
    cudaFree(first_ind_parent_g);
    cudaFree(activation_type_g);
    cudaFree(graph_n_g);
    cudaFree(graph_i_g);
    cudaFree(graph_p_g);
    cudaFree(graph_p_ind_n_g);
    cudaFree(graph_logic_g);
    cudaFree(bias_logic_g);
    cudaFree(dist_g);
    cudaFree(dist_input_g);
    cudaFree(weight_g);
    cudaFree(bias_g);
    cudaFree(weight_temp_g);
    cudaFree(bias_temp_g);
    cudaFree(weight_grad_g);
    cudaFree(bias_grad_g);
    cudaFree(mt_weight_g);
    cudaFree(mt_bias_g);
    cudaFree(ut_weight_g);
    cudaFree(ut_bias_g);
    cudaFree(vt_weight_g);
    cudaFree(vt_bias_g);
    cudaFree(mth_weight_g);
    cudaFree(mth_bias_g);
    cudaFree(vth_weight_g);
    cudaFree(vth_bias_g);

    // Deallocations --- graph
    deallocate_dmatrix(fix_weight_m, neuron_num);
    deallocate_dmatrix(fix_bias_m, neuron_num);
    deallocate_imatrix(activation_type_m, neuron_num);
    deallocate_imatrix(graph_p_m, all_input_num);
    deallocate_imatrix(graph_p_ind_n_m, all_input_num);

    deallocate_imatrix(graph_n_m, neuron_num);
    deallocate_imatrix(graph_i_m, neuron_num);
    deallocate_imatrix(graph_logic_m, neuron_num);
    deallocate_imatrix(bias_logic_m, neuron_num);
    deallocate_imatrix(parent_number_m, neuron_num);
    free(neighbour_number);
    free(bias_number);
    free(first_ind_neighbour);
    free(first_ind_bias);
    free(first_ind_parent);
    free(activation_type);
    free(graph_n);
    free(graph_i);
    free(graph_p);
    free(graph_p_ind_n);
    free(graph_logic);
    free(bias_logic);
    free(parent_number);
    free(fix_weight);
    free(fix_bias);
    free(graph_p_m_counter);

    // Deallocations shared weights and biases
    if (shared_weights_num > 0)
    {
        free(shared_weights_blocksize);
        deallocate_imatrix(shared_weights, shared_weights_num);
        deallocate_imatrix(shared_weights_indices, shared_weights_num);
        free(shared_weights_values);
    }
    if (shared_biases_num > 0)
    {
        free(shared_biases_blocksize);
        deallocate_imatrix(shared_biases, shared_biases_num);
        free(shared_biases_values);
    }

    // Deallocations --- weights, biases, gradients and momentums
    deallocate_dmatrix(weight_m, neuron_num);
    deallocate_dmatrix(bias_m, neuron_num);
    free(weight);
    free(bias);
    free(weight_best);
    free(bias_best);
    free(weight_grad);
    free(bias_grad);
    free(vt_weight);
    free(vt_bias);
    free(ut_weight);
    free(ut_bias);
    free(vth_weight);
    free(vth_bias);
    free(mt_weight);
    free(mt_bias);
    free(mth_weight);
    free(mth_bias);

    free(valid_labels);
    free(test_labels);

    free(test_pred_labels);
    // free(valid_pred_labels);

    free(errors_valid);
    free(errors_test);
    free(valid_fpr);
    free(valid_tpr);
    free(mcc_list);
    free(f1score_list);

    for (unsigned long long int i = 0; i < valid_num; i++)
    {
        free(predictions_valid[i]);
    }
    free(predictions_valid);
    for (unsigned long long int i = 0; i < test_num; i++)
    {
        free(predictions_test[i]);
    }
    free(predictions_test);

    // Deallocations --- PERT
    free(dist);
    free(dist_input);
    free(dist_extra);
    if ((ff_optimization > 0) && (check_cycle == 0))
    {
        free(dist_number);
        free(dist_number_temp);
        deallocate_imatrix(dist_indices_m, dist_max + 1);
        free(dist_indices);
        free(first_ind_dist_indices);
    }

    free(datas_mini_batch);
    free(train_data);
    free(valid_data);
    free(test_data);

    // Reset the CUDA device
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    return EXIT_SUCCESS;
}

float f_lr_momentum(unsigned long long int i, float start, float end, unsigned long long int T)
{
    return end + (start - end) * 0.5 * (1.0 + cosf(M_PI / (float)T * (float)i));
}

void read_parameters(char file_name[100])
{
    /**
     * Read the global variables from `file_name`
     */
    char temp_string[30];
    FILE *f = fopen(file_name, "r");
    if (f)
    {
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &seed);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &shuffle_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &thread_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &tol_fixit);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &maxiter_grad);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &maxiter_fix);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &initdx);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &sfreq);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", input_name);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", input_name_valid);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", input_name_test);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", output_name);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", predict_name_valid);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", predict_name_test);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", test_log);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", test_log_final);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &learn_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &valid_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &test_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &mini_batch_size);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &neuron_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &input_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &output_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &shared_weights_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &shared_biases_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", graph_datas);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", logic_datas);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", fixwb_datas);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", shared_w_datas);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", shared_b_datas);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &alpha);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", train_lossfunction_type);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", valid_metric_type);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", valid_metric_type_2);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &range_div);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &optimizer);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &nesterov);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &grad_alpha);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &adam_alpha);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &adam_beta1);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &adam_beta2);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &adam_eps);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &lr_scheduler);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &cyclic_momentum);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &pcr);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &base_momentum);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &max_momentum);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &div_factor);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &final_div_factor);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &step_size);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &lr_gamma);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &early_stopping);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &ff_optimization);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &clipping);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &clipping_threshold);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &loaddatas);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", load_backup);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", save_best_model);
        /*
        fscanf(f, "%s", temp_string);
        fscanf(f, "%llu", &numgrad);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &numgrad_eps);
        */
        fclose(f);
    }
    else
    {
        program_failure("File read error: simulparams.dat\n");
    }
}

void predict(float *train_data, float *valid_data, float *test_data,
             unsigned long long int mini_batch_num, unsigned long long int mini_batch_num_valid, unsigned long long int mini_batch_num_test,
             unsigned long long int *neighbour_number_g, unsigned long long int *neighbour_number, unsigned long long int *bias_number_g, unsigned long long int *bias_number, unsigned long long int *parent_number_g,
             unsigned long long int *first_ind_neighbour_g, unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias_g, unsigned long long int *first_ind_bias, unsigned long long int *first_ind_parent_g,
             unsigned long long int *activation_type_g, unsigned long long int *activation_type, unsigned long long int *graph_n_g, unsigned long long int *graph_n, unsigned long long int *graph_i_g, unsigned long long int *graph_i, unsigned long long int *graph_p_g,
             unsigned long long int *graph_p_ind_n_g,
             float *weight_g, float *weight_trans_g, float *bias_g,
             unsigned long long int dist_max,
             unsigned long long int *dist_g, unsigned long long int *dist,
             unsigned long long int *dist_input_g, unsigned long long int *dist_input,
             float **predictions_valid, float **predictions_test, float *errors_valid, float *errors_test)
{

    float *datas_mini_batch = (float *)malloc(mini_batch_size * (input_num + output_num) * sizeof(float));
    float **predictions_mini_batch = (float **)malloc(mini_batch_size * sizeof(float *));
    for (unsigned long long int i = 0; i < mini_batch_size; i++)
    {
        predictions_mini_batch[i] = (float *)malloc(output_num * sizeof(float));
    }

    // Training set
    /*
    for (unsigned long long int mini_batch_id = 0; mini_batch_id < mini_batch_num; mini_batch_id++)
    {
        unsigned long long int mini_batch_len;

        unsigned long long int mini_batch_si = mini_batch_id * mini_batch_size;
        unsigned long long int mini_batch_ei = (mini_batch_id + 1) * mini_batch_size - 1;
        if (mini_batch_ei > learn_num - 1)
        {
            mini_batch_ei = learn_num - 1;
        }
        mini_batch_len = mini_batch_ei - mini_batch_si + 1;

        for (unsigned long long int row_ind = mini_batch_si; row_ind <= mini_batch_ei; row_ind++)
        {
            for (unsigned long long int col_ind = 0; col_ind < input_num + output_num; col_ind++)
            {
                datas_mini_batch[(row_ind - mini_batch_si) * (input_num + output_num) + col_ind] = train_data[row_ind * (input_num + output_num) + col_ind];
            }
        }

        if (ff_optimization == 0)
        {
            make_predictions(datas_mini_batch, mini_batch_len,
                             neighbour_number_g, neighbour_number, bias_number_g, bias_number, parent_number_g,
                             first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias, first_ind_parent_g,
                             activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
                             graph_p_g, graph_p_ind_n_g, weight_trans_g, bias_g,
                             predictions_mini_batch);
        }
        else
        {
            make_predictions_ff(datas_mini_batch, mini_batch_len,
                                neighbour_number_g, neighbour_number, bias_number_g, bias_number, parent_number_g,
                                first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias, first_ind_parent_g,
                                activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
                                graph_p_g, graph_p_ind_n_g, weight_trans_g, bias_g,
                                dist_max,
                                dist_g, dist, dist_input_g, dist_input,
                                predictions_mini_batch);
        }

        if ((strcmp(train_lossfunction_type, "bce_multilabeling") == 0) || (strcmp(train_lossfunction_type, "multilabeling_crossentropy") == 0))
        {

            for (unsigned long long int i = 0; i < mini_batch_len; i++)
            {
                for (unsigned long long int j = 0; j < output_num; j++)
                {
                    *acc_learn += (unsigned long long int)roundf(predictions_mini_batch[i][j] + 0.01) == (unsigned long long int)roundf(datas_mini_batch[i * (input_num + output_num) + input_num + j] + 0.01);
                }
            }
        }

        if (strcmp(train_lossfunction_type, "multiclassification_crossentropy") == 0)
        {
            // Search the max ind in the predictions
            for (unsigned long long int i = 0; i < mini_batch_len; i++)
            {
                unsigned long long int pred_ind = 0;
                float pred_max = predictions_mini_batch[i][0];
                for (unsigned long long int j = 1; j < output_num; j++)
                {
                    if (predictions_mini_batch[i][j] > pred_max)
                    {
                        pred_ind = j;
                        pred_max = predictions_mini_batch[i][j];
                    }
                }

                // Search the max ind in the true classes
                unsigned long long int true_ind = 0;
                float true_max = datas_mini_batch[i * (input_num + output_num) + input_num + 0];
                for (unsigned long long int j = 1; j < output_num; j++)
                {
                    if (datas_mini_batch[i * (input_num + output_num) + input_num + j] > true_max)
                    {
                        true_ind = j;
                        true_max = datas_mini_batch[i * (input_num + output_num) + input_num + j];
                    }
                }

                *acc_learn += (float)pred_ind == true_ind;
            }
        }
    }
    */

    // Validation set
    unsigned long long int startind_err = 0;
    for (unsigned long long int mini_batch_id = 0; mini_batch_id < mini_batch_num_valid; mini_batch_id++)
    {
        unsigned long long int mini_batch_len;

        unsigned long long int mini_batch_si = mini_batch_id * mini_batch_size;
        unsigned long long int mini_batch_ei = (mini_batch_id + 1) * mini_batch_size - 1;
        if (mini_batch_ei > valid_num - 1)
        {
            mini_batch_ei = valid_num - 1;
        }
        mini_batch_len = mini_batch_ei - mini_batch_si + 1;

        for (unsigned long long int row_ind = mini_batch_si; row_ind <= mini_batch_ei; row_ind++)
        {
            for (unsigned long long int col_ind = 0; col_ind < input_num + output_num; col_ind++)
            {
                datas_mini_batch[(row_ind - mini_batch_si) * (input_num + output_num) + col_ind] = valid_data[row_ind * (input_num + output_num) + col_ind];
            }
        }

        if (ff_optimization == 0)
        {
            make_predictions(datas_mini_batch, mini_batch_len,
                             neighbour_number_g, neighbour_number, bias_number_g, bias_number, parent_number_g,
                             first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias, first_ind_parent_g,
                             activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
                             graph_p_g, graph_p_ind_n_g, weight_trans_g, bias_g,
                             predictions_mini_batch);
        }
        else
        {
            make_predictions_ff(datas_mini_batch, mini_batch_len,
                                neighbour_number_g, neighbour_number, bias_number_g, bias_number, parent_number_g,
                                first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias, first_ind_parent_g,
                                activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
                                graph_p_g, graph_p_ind_n_g, weight_trans_g, bias_g,
                                dist_max,
                                dist_g, dist, dist_input_g, dist_input,
                                predictions_mini_batch);
        }

        for (unsigned long long int i = 0; i < mini_batch_len; i++)
        {
            for (unsigned long long int j = 0; j < output_num; j++)
            {
                predictions_valid[i + mini_batch_size * mini_batch_id][j] = predictions_mini_batch[i][j];
            }

            errors_valid[i + mini_batch_size * mini_batch_id] = 0.0;

            if (strcmp(train_lossfunction_type, "MSE") == 0)
            {
                for (unsigned long long int j = 0; j < output_num; j++)
                {

                    errors_valid[i + mini_batch_size * mini_batch_id] += powf(predictions_mini_batch[i][j] - datas_mini_batch[i * (input_num + output_num) + j + input_num], 2.0); // + input_num
                }
                errors_valid[i + mini_batch_size * mini_batch_id] /= output_num;
                errors_valid[i + mini_batch_size * mini_batch_id] = sqrtf(errors_valid[i + mini_batch_size * mini_batch_id]);
            }

            if (strcmp(train_lossfunction_type, "MAE") == 0)
            {
                for (unsigned long long int j = 0; j < output_num; j++)
                {

                    errors_valid[i + mini_batch_size * mini_batch_id] += fabsf(predictions_mini_batch[i][j] - datas_mini_batch[i * (input_num + output_num) + j + input_num]); // + input_num
                }
                errors_valid[i + mini_batch_size * mini_batch_id] /= output_num;
            }
        }

        /*
        if ((strcmp(train_lossfunction_type, "bce_multilabeling") == 0) || (strcmp(train_lossfunction_type, "multilabeling_crossentropy") == 0))
        {

            for (unsigned long long int i = 0; i < mini_batch_len; i++)
            {
                for (unsigned long long int j = 0; j < output_num; j++)
                {
                    *acc_valid += (unsigned long long int)roundf(predictions_mini_batch[i][j] + 0.01) == (unsigned long long int)roundf(datas_mini_batch[i * (input_num + output_num) + input_num + j] + 0.01);
                }
            }
        }

        if (strcmp(train_lossfunction_type, "multiclassification_crossentropy") == 0)
        {
            // Search the max ind in the predictions
            for (unsigned long long int i = 0; i < mini_batch_len; i++)
            {
                unsigned long long int pred_ind = 0;
                float pred_max = predictions_mini_batch[i][0];
                for (unsigned long long int j = 1; j < output_num; j++)
                {
                    if (predictions_mini_batch[i][j] > pred_max)
                    {
                        pred_ind = j;
                        pred_max = predictions_mini_batch[i][j];
                    }
                }

                // Search the max ind in the true classes
                unsigned long long int true_ind = 0;
                float true_max = datas_mini_batch[i * (input_num + output_num) + input_num + 0];
                for (unsigned long long int j = 1; j < output_num; j++)
                {
                    if (datas_mini_batch[i * (input_num + output_num) + input_num + j] > true_max)
                    {
                        true_ind = j;
                        true_max = datas_mini_batch[i * (input_num + output_num) + input_num + j];
                    }
                }

                *acc_valid += (float)pred_ind == true_ind;
            }
        }
        */
        startind_err += mini_batch_len;
    }

    // Test set
    startind_err = 0;
    for (unsigned long long int mini_batch_id = 0; mini_batch_id < mini_batch_num_test; mini_batch_id++)
    {
        unsigned long long int mini_batch_len;

        unsigned long long int mini_batch_si = mini_batch_id * mini_batch_size;
        unsigned long long int mini_batch_ei = (mini_batch_id + 1) * mini_batch_size - 1;
        if (mini_batch_ei > test_num - 1)
        {
            mini_batch_ei = test_num - 1;
        }
        mini_batch_len = mini_batch_ei - mini_batch_si + 1;

        for (unsigned long long int row_ind = mini_batch_si; row_ind <= mini_batch_ei; row_ind++)
        {
            for (unsigned long long int col_ind = 0; col_ind < input_num + output_num; col_ind++)
            {
                datas_mini_batch[(row_ind - mini_batch_si) * (input_num + output_num) + col_ind] = test_data[row_ind * (input_num + output_num) + col_ind];
            }
        }

        if (ff_optimization == 0)
        {
            make_predictions(datas_mini_batch, mini_batch_len,
                             neighbour_number_g, neighbour_number, bias_number_g, bias_number, parent_number_g,
                             first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias, first_ind_parent_g,
                             activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
                             graph_p_g, graph_p_ind_n_g, weight_trans_g, bias_g,
                             predictions_mini_batch);
        }
        else
        {
            make_predictions_ff(datas_mini_batch, mini_batch_len,
                                neighbour_number_g, neighbour_number, bias_number_g, bias_number, parent_number_g,
                                first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias, first_ind_parent_g,
                                activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
                                graph_p_g, graph_p_ind_n_g, weight_trans_g, bias_g,
                                dist_max,
                                dist_g, dist, dist_input_g, dist_input,
                                predictions_mini_batch);
        }

        for (unsigned long long int i = 0; i < mini_batch_len; i++)
        {
            for (unsigned long long int j = 0; j < output_num; j++)
            {
                predictions_test[i + mini_batch_size * mini_batch_id][j] = predictions_mini_batch[i][j];
            }

            errors_test[i + mini_batch_size * mini_batch_id] = 0.0;

            if (strcmp(train_lossfunction_type, "MSE") == 0)
            {
                for (unsigned long long int j = 0; j < output_num; j++)
                {

                    errors_test[i + mini_batch_size * mini_batch_id] += powf(predictions_mini_batch[i][j] - datas_mini_batch[i * (input_num + output_num) + j + input_num], 2.0); // + input_num
                }
                errors_test[i + mini_batch_size * mini_batch_id] /= output_num;
                errors_test[i + mini_batch_size * mini_batch_id] = sqrtf(errors_test[i + mini_batch_size * mini_batch_id]);
            }

            if (strcmp(train_lossfunction_type, "MAE") == 0)
            {
                for (unsigned long long int j = 0; j < output_num; j++)
                {

                    errors_test[i + mini_batch_size * mini_batch_id] += fabsf(predictions_mini_batch[i][j] - datas_mini_batch[i * (input_num + output_num) + j + input_num]); // + input_num
                }
                errors_test[i + mini_batch_size * mini_batch_id] /= output_num;
            }
        }

        /*
        if ((strcmp(train_lossfunction_type, "bce_multilabeling") == 0) || (strcmp(train_lossfunction_type, "multilabeling_crossentropy") == 0))
        {

            for (unsigned long long int i = 0; i < mini_batch_len; i++)
            {
                for (unsigned long long int j = 0; j < output_num; j++)
                {
                    *acc_test += (unsigned long long int)roundf(predictions_mini_batch[i][j] + 0.01) == (unsigned long long int)roundf(datas_mini_batch[i * (input_num + output_num) + input_num + j] + 0.01);
                }
            }
        }

        if (strcmp(train_lossfunction_type, "multiclassification_crossentropy") == 0)
        {
            // Search the max ind in the predictions
            for (unsigned long long int i = 0; i < mini_batch_len; i++)
            {
                unsigned long long int pred_ind = 0;
                float pred_max = predictions_mini_batch[i][0];
                for (unsigned long long int j = 1; j < output_num; j++)
                {
                    if (predictions_mini_batch[i][j] > pred_max)
                    {
                        pred_ind = j;
                        pred_max = predictions_mini_batch[i][j];
                    }
                }

                // Search the max ind in the true classes
                unsigned long long int true_ind = 0;
                float true_max = datas_mini_batch[i * (input_num + output_num) + input_num + 0];
                for (unsigned long long int j = 1; j < output_num; j++)
                {
                    if (datas_mini_batch[i * (input_num + output_num) + input_num + j] > true_max)
                    {
                        true_ind = j;
                        true_max = datas_mini_batch[i * (input_num + output_num) + input_num + j];
                    }
                }

                *acc_test += (float)pred_ind == true_ind;
            }
        }
        */
        startind_err += mini_batch_len;
    }

    free(datas_mini_batch);
    for (unsigned long long int i = 0; i < mini_batch_size; i++)
    {
        free(predictions_mini_batch[i]);
    }
    free(predictions_mini_batch);
}

void read_graph(char graph_file_name[100], char logic_file_name[100], char fixwb_file_name[100],
                char shared_w_file_name[100], char shared_b_file_name[100],
                unsigned long long int *neighbour_number, unsigned long long int *bias_number,
                unsigned long long int *shared_weights_blocksize, unsigned long long int *shared_biases_blocksize,
                unsigned long long int **shared_weights_indices,
                unsigned long long int **shared_weights, unsigned long long int **shared_biases,
                unsigned long long int **activation_type, unsigned long long int **graph_n, unsigned long long int **graph_i,
                unsigned long long int **graph_logic, unsigned long long int **bias_logic, unsigned long long int **parent_number,
                float **fix_weight, float **fix_bias,
                unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias)
{
    /**
     * Read graph data, allocate memory
     */
    char temp_string[30];

    FILE *f_graph = fopen(graph_file_name, "r");
    FILE *f_logic = fopen(logic_file_name, "r");
    FILE *f_fixwb = fopen(fixwb_file_name, "r");
    if (f_graph && f_logic && f_fixwb)
    {
        // Read the graph
        for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            fscanf(f_graph, "%llu", &neighbour_number[neuron_id]);
            fscanf(f_graph, "%s", temp_string);

            if (neuron_id > 0)
            {
                first_ind_neighbour[neuron_id] = first_ind_neighbour[neuron_id - 1] + neighbour_number[neuron_id - 1];
            }
            all_neighbour_num += neighbour_number[neuron_id];

            graph_n[neuron_id] = (unsigned long long int *)malloc(neighbour_number[neuron_id] * sizeof(unsigned long long int));
            graph_i[neuron_id] = (unsigned long long int *)malloc(neighbour_number[neuron_id] * sizeof(unsigned long long int));

            graph_logic[neuron_id] = (unsigned long long int *)malloc(neighbour_number[neuron_id] * sizeof(unsigned long long int));
            fix_weight[neuron_id] = (float *)malloc(neighbour_number[neuron_id] * sizeof(float));

            for (unsigned long long int i = 0; i < neighbour_number[neuron_id]; i++)
            {
                fscanf(f_graph, "%llu", &graph_n[neuron_id][i]);
                graph_n[neuron_id][i]--;
                fscanf(f_graph, "%llu", &graph_i[neuron_id][i]);
                graph_i[neuron_id][i]--;
                fscanf(f_graph, "%s", temp_string);
            }

            for (unsigned long long int i = 0; i < neighbour_number[neuron_id]; i++)
            {
                fscanf(f_logic, "%llu", &graph_logic[neuron_id][i]);
                if (graph_logic[neuron_id][i] == 0)
                {
                    fscanf(f_fixwb, "%f", &fix_weight[neuron_id][i]);
                }
            }

            fscanf(f_graph, "%s", temp_string);
            fscanf(f_logic, "%s", temp_string);
            fscanf(f_fixwb, "%s", temp_string);

            fscanf(f_graph, "%llu", &bias_number[neuron_id]);
            fscanf(f_graph, "%s", temp_string);

            if (neuron_id > 0)
            {
                first_ind_bias[neuron_id] = first_ind_bias[neuron_id - 1] + bias_number[neuron_id - 1];
            }
            all_input_num += bias_number[neuron_id];

            activation_type[neuron_id] = (unsigned long long int *)malloc(bias_number[neuron_id] * sizeof(unsigned long long int));
            bias_logic[neuron_id] = (unsigned long long int *)malloc(bias_number[neuron_id] * sizeof(unsigned long long int));
            fix_bias[neuron_id] = (float *)malloc(bias_number[neuron_id] * sizeof(float));
            for (unsigned long long int i = 0; i < bias_number[neuron_id]; i++)
            {
                fscanf(f_graph, "%llu", &activation_type[neuron_id][i]);
            }

            for (unsigned long long int i = 0; i < bias_number[neuron_id]; i++)
            {
                fscanf(f_logic, "%llu", &bias_logic[neuron_id][i]);
                if (bias_logic[neuron_id][i] == 0)
                {
                    fscanf(f_fixwb, "%f", &fix_bias[neuron_id][i]);
                }
            }
        }

        // Calculate the numbers of the parents
        for (unsigned long long int i = 0; i < neuron_num; i++)
        {
            parent_number[i] = (unsigned long long int *)malloc((bias_number[i]) * sizeof(unsigned long long int));
            for (unsigned long long int j = 0; j < bias_number[i]; j++)
            {
                parent_number[i][j] = 0;
            }
        };
        for (unsigned long long int i = 0; i < neuron_num; i++)
        {
            for (unsigned long long int j = 0; j < neighbour_number[i]; j++)
            {
                parent_number[graph_n[i][j]][graph_i[i][j]]++;
            }
        }
    }
    else
    {
        program_failure("File read error in graph files!");
    }
    fclose(f_graph);
    fclose(f_logic);
    fclose(f_fixwb);

    // Shared weights
    // printf("Shared weight num: %llu\n ", shared_weights_num);

    if (shared_weights_num > 0)
    {
        FILE *f_shared_weights = fopen(shared_w_file_name, "r"); // fajlnevet majd modositani
        if (f_shared_weights)
        {
            for (unsigned long long int group_id = 0; group_id < shared_weights_num; group_id++)
            {
                fscanf(f_shared_weights, "%llu", &shared_weights_blocksize[group_id]); // ezt majd definialni kell
                fscanf(f_shared_weights, "%s", temp_string);

                shared_weights[group_id] = (unsigned long long int *)malloc(3 * shared_weights_blocksize[group_id] * sizeof(unsigned long long int));
                shared_weights_indices[group_id] = (unsigned long long int *)malloc(shared_weights_blocksize[group_id] * sizeof(unsigned long long int));

                for (unsigned long long int i = 0; i < shared_weights_blocksize[group_id]; i++)
                {
                    fscanf(f_shared_weights, "%llu", &shared_weights[group_id][i * 3]);
                    shared_weights[group_id][i * 3]--;
                    fscanf(f_shared_weights, "%llu", &shared_weights[group_id][i * 3 + 1]);
                    shared_weights[group_id][i * 3 + 1]--;
                    fscanf(f_shared_weights, "%llu", &shared_weights[group_id][i * 3 + 2]);
                    shared_weights[group_id][i * 3 + 2]--;
                    fscanf(f_shared_weights, "%s", temp_string);
                }
            }
            fclose(f_shared_weights);

            // shared weight indices
            for (unsigned long long int group_id = 0; group_id < shared_weights_num; group_id++)
            {
                for (unsigned long long int weight_id = 0; weight_id < shared_weights_blocksize[group_id]; weight_id++)
                {
                    unsigned long long int j = 0;

                    /*
                    for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++){
                        for (unsigned long long int neighbour_id = 0; neighbour_id < neighbour_number[neuron_id]; neighbour_id++){

                        }
                    }
                    */

                    while ((shared_weights[group_id][weight_id * 3 + 1] != graph_n[shared_weights[group_id][weight_id * 3]][j]) ||
                           (shared_weights[group_id][weight_id * 3 + 2] != graph_i[shared_weights[group_id][weight_id * 3]][j]))
                    {
                        j++;
                        if (j >= neighbour_number[shared_weights[group_id][weight_id * 3]])
                        {
                            program_failure("Shared weight index problem!");
                        }
                    }

                    // weight[  shared_weights[group_id][weight_id * 3]  ] [ j ] <--- ezt a j-t meg kell talalni es el is
                    // kellene menteni valahova; j azt tudja hogy:
                    // - shared_weights[group_id][weight_id * 3 + 1] == graph_n[ shared_weights[group_id][weight_id * 3] ][j]
                    // - shared_weights[group_id][weight_id * 3 + 2] == graph_i[ shared_weights[group_id][weight_id * 3] ][j]
                    //
                    // shared_weight_indices --- shared_weights_num * shared_weights_blocksize --- ugyanez bias-re
                    // shared_weights_indices[group_id][weight_id] = shared_weights[group_id][weight_id * 3];
                    shared_weights_indices[group_id][weight_id] = j;
                    // printf(" %llu ",j);
                }
            }
            /*
            for (unsigned long long int group_id = 0; group_id < shared_weights_num; group_id++)
            {
                printf(" %llu %llu ||| ", group_id, shared_weights_blocksize[group_id]);
                for (unsigned long long int weight_id = 0; weight_id < shared_weights_blocksize[group_id]; weight_id++)
                {

                    printf(" %llu %llu | ", shared_weights_indices[group_id][2*weight_id]+1, shared_weights_indices[group_id][2*weight_id+1]+1);
                }
                printf("\n");
            }
            printf("\n");
            */
        }
        else
        {
            program_failure("File read error in shared weight file!");
        }
    }

    // Shared biases

    // printf("Shared bias num: %llu\n ", shared_biases_num);
    if (shared_biases_num > 0)
    {
        FILE *f_shared_biases = fopen(shared_b_file_name, "r"); // fajlnevet majd modositani
        if (f_shared_biases)
        {
            for (unsigned long long int group_id = 0; group_id < shared_biases_num; group_id++)
            {
                fscanf(f_shared_biases, "%llu", &shared_biases_blocksize[group_id]); // ezt majd definialni kell
                fscanf(f_shared_biases, "%s", temp_string);

                shared_biases[group_id] = (unsigned long long int *)malloc(2 * shared_biases_blocksize[group_id] * sizeof(unsigned long long int));

                for (unsigned long long int i = 0; i < shared_biases_blocksize[group_id]; i++)
                {
                    fscanf(f_shared_biases, "%llu", &shared_biases[group_id][i * 2]);
                    shared_biases[group_id][i * 2]--;
                    fscanf(f_shared_biases, "%llu", &shared_biases[group_id][i * 2 + 1]);
                    shared_biases[group_id][i * 2 + 1]--;
                    fscanf(f_shared_biases, "%s", temp_string);
                }
            }
            fclose(f_shared_biases);
        }
        else
        {
            program_failure("File read error in shared bias file!");
        }
    }
}

void program_failure(char str[])
{
    /**
     * Program failure
     */
    perror(str);
    exit(EXIT_FAILURE);
}

void read_data(float *datas, unsigned long long int line_number, FILE *f_data, unsigned long long int test)
{
    /**
     * Read the data
     */
    if (f_data)
    {
        unsigned long long int output_num_temp = 0;
        if (test == 0)
        {
            output_num_temp = output_num;
        }

        for (unsigned long long int i = 0; i < line_number; i++)
        {
            for (unsigned long long int j = 0; j < input_num + output_num_temp; j++)
            {
                fscanf(f_data, "%f", &datas[i * (input_num + output_num) + j]);
            }
        }
    }
    else
    {
        program_failure("File read error in data file!");
    }
}

unsigned long long int rand_range_int(unsigned long long int min, unsigned long long int max)
{
    /**
     * Generates a random integer between min and max
     */
    return rand() % (max - min + 1) + min;
}

float rand_range(float min, float max)
{
    /**
     * Generates a random float number between min and max
     */

    return min + (float)rand() / RAND_MAX * (max - min);
}

float act_fun(float x, unsigned long long int chooser)
{
    /**
     * Calculate the activation function type `chooser` on the input `x`
     */
    switch (chooser)
    {
    case 0:
        return x;
        break;
    case 1:
        return 1.0 / (1.0 + expf(-x));
        break;
    case 2:
        return tanhf(x);
        // if (x>0){
        //     return x/(1.0+x);
        // }
        // else
        //{
        //     return x/(1.0-x);
        // }
        break;
    case 3:
        if (x > 0)
        {
            return x;
        }
        else
        {
            return 0.0;
        }
        break;
    case 4:
        return x / (1.0 + expf(-x));
        break;
    case 6:
        return 1.0 - x;
        break;
    case 7:
        return 1.0 / x;
        break;
    case 8:
        return cosf(x);
        break;
    case 9:
        return atanf(x);
        break;
    case 10:
        if (x > 0)
        {
            return x;
        }
        else
        {
            return 0.1 * (expf(x) - 1.0);
        }
        break;
    case 11:

        // return atanf(x)/(1.0+x*x);
        return logf(sqrtf(x * x + 1.0) + x);

        break;
    default:
        return 0.0;
        break;
    }
}

float act_fun_diff(float x, unsigned long long int chooser)
{
    /**
     * Calculate the derivative of the activation function type `chooser` on the input `x`
     */
    switch (chooser)
    {
    case 0:
        return 1.0;
        break;
    case 1:
        return act_fun(x, chooser) * (1.0 - act_fun(x, chooser));
        break;
    case 2:
        return 1.0 - tanhf(x) * tanhf(x);
        // if (x>0){
        //     return 1.0/((1.0+x)*(1.0+x));
        // }
        // else
        //{
        //     return 1.0/((1.0-x)*(1.0-x));
        // }
        break;
    case 3:
        if (x > 0)
        {
            return 1.0;
        }
        else
        {
            return 0.0;
        }
        break;
    case 4:
        return (1.0 + expf(-x) + x * expf(-x)) / powf(1.0 + expf(-x), 2.0);
        break;
    case 6:
        return -1.0;
        break;
    case 7:
        return -1.0 / powf(x, 2.0);
        break;
    case 8:
        return -sinf(x);
        break;
    case 9:
        return 1.0 / (1.0 + x * x);
        break;
    case 10:
        if (x > 0)
        {
            return 1;
        }
        else
        {
            return 0.1 * expf(x);
        }
        break;
    case 11:

        // return (1.0-2.0*x*atanf(x))/powf(1.0+x*x,2.0);
        return 1.0 / sqrtf(x * x + 1.0);

        break;
    default:
        return 0.0;
        break;
    }
}

float calc_error(float *neuron_value, float *target_vector, unsigned long long int mini_batch_len)
{
    /*
    Calculating the error functions
    */

    if (strcmp(train_lossfunction_type, "MSE") == 0)
    {

        float returner = 0.0;
        for (unsigned long long int data_index = 0; data_index < mini_batch_len; data_index++)
        {
            for (unsigned long long int i = 0; i < output_num; i++)
            {
                unsigned long long int neuron_id = neuron_num - output_num + i;
                returner += powf((neuron_value[data_index * neuron_num + neuron_id] - target_vector[data_index * output_num + i]), 2);
            }
        }
        return returner;
    }
    if (strcmp(train_lossfunction_type, "MAE") == 0)
    {

        float returner = 0.0;
        for (unsigned long long int data_index = 0; data_index < mini_batch_len; data_index++)
        {
            for (unsigned long long int i = 0; i < output_num; i++)
            {
                unsigned long long int neuron_id = neuron_num - output_num + i;
                returner += fabsf(neuron_value[data_index * neuron_num + neuron_id] - target_vector[data_index * output_num + i]);
            }
        }
        return returner;
    }
    if (strcmp(train_lossfunction_type, "multilabeling_crossentropy") == 0)
    {
        float returner = 0.0;

        for (unsigned long long int data_index = 0; data_index < mini_batch_len; data_index++)
        {
            for (unsigned long long int i = 0; i < output_num; i++)
            {
                unsigned long long int neuron_id = neuron_num - output_num + i;
                if ((neuron_value[data_index * neuron_num + neuron_id] > 0.0) && (neuron_value[data_index * neuron_num + neuron_id] < 1.0))
                {
                    returner -=
                        target_vector[data_index * output_num + i] * logf(neuron_value[data_index * neuron_num + neuron_id]) +
                        (1.0 - target_vector[data_index * output_num + i]) * logf(1.0 - neuron_value[data_index * neuron_num + neuron_id]);
                }
            }
        }
        return returner;
    }

    if (strcmp(train_lossfunction_type, "bce_multilabeling") == 0)
    {

        float returner = 0.0;
        for (unsigned long long int data_index = 0; data_index < mini_batch_len; data_index++)
        {
            for (unsigned long long int i = 0; i < output_num; i++)
            {
                unsigned long long int neuron_id = neuron_num - output_num + i;
                returner +=
                    (1.0 - target_vector[data_index * output_num + i]) * neuron_value[data_index * neuron_num + neuron_id] +
                    logf(1.0 + expf(-neuron_value[data_index * neuron_num + neuron_id]));
            }
        }
        return returner;
    }

    if (strcmp(train_lossfunction_type, "multiclassification_crossentropy") == 0)
    {
        float returner = 0.0;
        for (unsigned long long int data_index = 0; data_index < mini_batch_len; data_index++)
        {

            float *softmax_vec = (float *)malloc(output_num * sizeof(float));

            float sum_softmax = 0.0;
            for (unsigned long long int i = 0; i < output_num; i++)
            {
                unsigned long long int neuron_id = neuron_num - output_num + i;
                softmax_vec[i] = neuron_value[data_index * neuron_num + neuron_id];
                sum_softmax += expf(softmax_vec[i]);
            }

            // softmax(softmax_vec, output_num);

            for (unsigned long long int i = 0; i < output_num; i++)
            {
                returner -= target_vector[data_index * output_num + i] * (softmax_vec[i] - logf(sum_softmax));
            }

            free(softmax_vec);
        }
        return returner;
    }
}

unsigned long long int imax(unsigned long long int a, unsigned long long int b)
{
    /**
     * Returns max(a,b) --- integer
     */
    if (a > b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

void copy_dmatrix(float **input_matrix, unsigned long long int row_num, unsigned long long int *col_num, float *output_matrix)
{
    unsigned long long int ind = 0;
    for (unsigned long long int i = 0; i < row_num; i++)
    {
        for (unsigned long long int j = 0; j < col_num[i]; j++)
        {
            output_matrix[ind] = input_matrix[i][j];
            ind++;
        }
    }
}
void copy_imatrix(unsigned long long int **input_matrix, unsigned long long int row_num, unsigned long long int *col_num, unsigned long long int *output_matrix)
{
    unsigned long long int ind = 0;
    for (unsigned long long int i = 0; i < row_num; i++)
    {
        for (unsigned long long int j = 0; j < col_num[i]; j++)
        {
            output_matrix[ind] = input_matrix[i][j];
            ind++;
        }
    }
}

float **allocate_dmatrix(unsigned long long int row_num, unsigned long long int *col_num)
{
    float **returner = (float **)malloc(row_num * sizeof(float *));
    for (unsigned long long int i = 0; i < row_num; i++)
    {
        returner[i] = (float *)malloc(col_num[i] * sizeof(float));
    }
    return returner;
}

unsigned long long int **allocate_imatrix(unsigned long long int row_num, unsigned long long int *col_num)
{
    unsigned long long int **returner = (unsigned long long int **)malloc(row_num * sizeof(unsigned long long int *));
    for (unsigned long long int i = 0; i < row_num; i++)
    {
        returner[i] = (unsigned long long int *)malloc(col_num[i] * sizeof(unsigned long long int));
    }
    return returner;
}

void deallocate_dmatrix(float **m, unsigned long long int row_num)
{
    for (unsigned long long int i = 0; i < row_num; i++)
    {
        free(m[i]);
    }
    free(m);
}

void deallocate_imatrix(unsigned long long int **m, unsigned long long int row_num)
{
    for (unsigned long long int i = 0; i < row_num; i++)
    {
        free(m[i]);
    }
    free(m);
}

void print_progress_bar(unsigned long long int max_length, float rate)
{

    printf("[");
    unsigned long long int act_length = round(max_length * rate);
    for (unsigned long long int i = 0; i < act_length; i++)
    {
        printf("=");
    }
    printf(">");
    for (unsigned long long int i = 0; i < max_length - act_length; i++)
    {
        printf(".");
    }
    printf("] ");
}

//===========================================================================
//=  Function to generate normally distributed random variable using the    =
//=  Box-Muller method                                                      =
//=    - Input: mean and standard deviation                                 =
//=    - Output: Returns with normally distributed random variable          =
//===========================================================================

float random_normal(float mean, float std_dev)
{
    float u, r, theta; // Variables for Box-Muller method
    float x;           // Normal(0, 1) rv
    float norm_rv;     // The adjusted normal rv

    // Generate u
    u = 0.0;
    while (u == 0.0)
        u = rand_range(0.0, 1.0);

    // Compute r
    r = sqrtf(-2.0 * logf(u));

    // Generate theta
    theta = 0.0;
    while (theta == 0.0)
        theta = 2.0 * M_PI * rand_range(0.0, 1.0);

    // Generate x value
    x = r * cosf(theta);

    // Adjust x value for specified mean and variance
    norm_rv = (x * std_dev) + mean;

    // Return the normally distributed RV value
    return (norm_rv);
}

void softmax(float *input, unsigned long long int input_len)
{
    //    assert (input != NULL);
    //    assert (input_len != 0);
    unsigned long long int i;
    float m;
    /* Find maximum value from input array */
    m = input[0];
    for (i = 1; i < input_len; i++)
    {
        if (input[i] > m)
        {
            m = input[i];
        }
    }

    float sum = 0;
    for (i = 0; i < input_len; i++)
    {
        sum += expf(input[i] - m);
    }

    for (i = 0; i < input_len; i++)
    {
        input[i] = expf(input[i] - m - logf(sum));
    }
}

void initialize_weights(unsigned long long int *neighbour_number, unsigned long long int *bias_number,
                        unsigned long long int *shared_weights_blocksize, unsigned long long int *shared_biases_blocksize,
                        unsigned long long int **shared_weights_indices,
                        unsigned long long int **shared_weights, unsigned long long int **shared_biases,
                        unsigned long long int **activation_type, unsigned long long int **graph_n, unsigned long long int **graph_i,
                        unsigned long long int **parent_number, float **weight, float **bias)
{
    /**
     *  Initialize the weights and the biases
     */
    float *shared_weights_values_init, *shared_biases_values_init;

    if (shared_weights_num > 0)
    {
        shared_weights_values_init = (float *)calloc(shared_weights_num, sizeof(float));
    }
    if (shared_biases_num > 0)
    {
        shared_biases_values_init = (float *)calloc(shared_biases_num, sizeof(float));
    }

    // https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
    // Initialize the weights
    for (unsigned long long int i = 0; i < neuron_num; i++)
    {
        weight[i] = (float *)malloc((neighbour_number[i]) * sizeof(float));
        for (unsigned long long int j = 0; j < neighbour_number[i]; j++)
        {
            // weight[i][j] = rand_range(-initdx, initdx) / (float)(parent_number[graph_n[i][j]][graph_i[i][j]] + 1.0);
            // weight[i][j] = rand_range(-initdx, initdx) * sqrtf(2.0) / sqrtf((float)(parent_number[graph_n[i][j]][graph_i[i][j]] + neighbour_number[i]));
            // weight[i][j] = random_normal(0.0, 1.0) * sqrtf(initdx * 2.0 / (float)(parent_number[graph_n[i][j]][graph_i[i][j]]));
            // weight[i][j] = random_normal(0.0, 1.0) * sqrtf(initdx * 2.0 / (float)(parent_number[graph_n[i][j]][graph_i[i][j]] + neighbour_number[i]));
            weight[i][j] = rand_range(-initdx, initdx) / sqrtf((float)(parent_number[graph_n[i][j]][graph_i[i][j]])); // Torch style: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L44-L48
            // Also https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
            // weight[i][j] = rand_range(-initdx, initdx) * sqrtf(2.0) / sqrtf((float)(parent_number[graph_n[i][j]][graph_i[i][j]] + neighbour_number[i]));
        }
    };

    // Initialize the bias
    for (unsigned long long int i = 0; i < neuron_num; i++)
    {
        bias[i] = (float *)malloc((bias_number[i]) * sizeof(float));
        for (unsigned long long int j = 0; j < bias_number[i]; j++)
        {
            bias[i][j] = 0.0;

            if (parent_number[i][j] > 0)
            {
                bias[i][j] = rand_range(-initdx, initdx) / sqrtf((float)(parent_number[i][j])); // Torch style: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L44-L48
                // Also https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
                // bias[i][j] = rand_range(-initdx, initdx) * sqrtf(2.0) / sqrtf((float)(parent_number[i][j] + neighbour_number[i]));
                // bias[i][j] = random_normal(0.0, 1.0) * sqrtf(initdx * 2.0 )  / sqrtf((float)(parent_number[i][j]));
            }
        }
    }

    if (shared_weights_num > 0)
    {
        // Colllect the shared weight values
        for (unsigned long long int group_id = 0; group_id < shared_weights_num; group_id++)
        {
            for (unsigned long long int weight_id = 0; weight_id < shared_weights_blocksize[group_id]; weight_id++)
            {
                // weight[  shared_weights[group_id][weight_id * 3]  ] [ j ] <--- ezt a j-t meg kell talalni es el is
                // kellene menteni valahova; j azt tudja hogy:
                // - shared_weights[group_id][weight_id * 3 + 1] == graph_n[ shared_weights[group_id][weight_id * 3] ][j]
                // - shared_weights[group_id][weight_id * 3 + 2] == graph_i[ shared_weights[group_id][weight_id * 3] ][j]
                //
                // shared_weight_indices --- shared_weights_num * shared_weights_blocksize --- ugyanez bias-re
                //
                shared_weights_values_init[group_id] += weight[shared_weights[group_id][weight_id * 3]][shared_weights_indices[group_id][weight_id]];
            }
            shared_weights_values_init[group_id] /= shared_weights_blocksize[group_id];
        }

        // distribute the shared weights
        for (unsigned long long int group_id = 0; group_id < shared_weights_num; group_id++)
        {
            for (unsigned long long int weight_id = 0; weight_id < shared_weights_blocksize[group_id]; weight_id++)
            {
                unsigned long long int i = shared_weights[group_id][weight_id * 3];
                unsigned long long int j = shared_weights_indices[group_id][weight_id];
                weight[i][j] = shared_weights_values_init[group_id];
            }
        }
    }

    if (shared_biases_num > 0)
    {
        // Collect the shared bias values
        for (unsigned long long int group_id = 0; group_id < shared_biases_num; group_id++)
        {
            for (unsigned long long int bias_id = 0; bias_id < shared_biases_blocksize[group_id]; bias_id++)
            {
                shared_biases_values_init[group_id] += bias[shared_biases[group_id][bias_id * 2]][shared_biases[group_id][bias_id * 2 + 1]];
            }
            shared_biases_values_init[group_id] /= shared_biases_blocksize[group_id];
        }

        // distribute the shared biases
        for (unsigned long long int group_id = 0; group_id < shared_biases_num; group_id++)
        {
            for (unsigned long long int bias_id = 0; bias_id < shared_biases_blocksize[group_id]; bias_id++)
            {
                bias[shared_biases[group_id][bias_id * 2]][shared_biases[group_id][bias_id * 2 + 1]] = shared_biases_values_init[group_id];
            }
        }
    }
    /*
    for (unsigned long long int i = 0; i < neuron_num; i++)
    {
        printf("%llu | ",i+1);
        for (unsigned long long int j = 0; j < neighbour_number[i]; j++)
        {
            printf(" %.5f ", weight[i][j]);
        }
        printf("\n");
    }
    */

    if (shared_weights_num > 0)
    {
        free(shared_weights_values_init);
    }
    if (shared_biases_num > 0)
    {
        free(shared_biases_values_init);
    }
}

float calc_gradient_mini_batch(float *datas, unsigned long long int mini_batch_len,
                               unsigned long long int *neighbour_number_g, unsigned long long int *neighbour_number, unsigned long long int *bias_number_g, unsigned long long int *bias_number, unsigned long long int *parent_number_g,
                               unsigned long long int *first_ind_neighbour_g, unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias_g, unsigned long long int *first_ind_bias, unsigned long long int *first_ind_parent_g,
                               unsigned long long int *activation_type_g, unsigned long long int *activation_type, unsigned long long int *graph_n_g, unsigned long long int *graph_n, unsigned long long int *graph_i_g, unsigned long long int *graph_i, unsigned long long int *graph_p_g,
                               unsigned long long int *graph_p_ind_n_g,
                               float *weight_g, float *bias_g,
                               float *weight_grad_g, float *bias_grad_g,
                               float *iter_forward, float *iter_backward)
{
    // Calculating the gradient on a mini-batch

    // Definitions
    float error_mini_batch = 0.0;
    float iter_forward_temp = 0.0;
    float iter_backward_temp = 0.0;
    unsigned long long int nthreads;

    // Reset gradients
    set_zero_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_grad_g, all_neighbour_num);
    set_zero_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(bias_grad_g, all_input_num);

    //++++++++++++++++++++++++++//
    //                          //
    // Loop over the mini-batch //
    //                          //
    //++++++++++++++++++++++++++//

    // Loop over the elements on the elements of the mini-batch

    float error_temp;
    unsigned long long int iter_f, iter_b;
    float error_iter;

    float *input_value = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *neuron_value = (float *)calloc(mini_batch_len * neuron_num, sizeof(float));
    float *target_vector = (float *)calloc(mini_batch_len * output_num, sizeof(float));
    float *input_value_old = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *input_value_orig = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *weight_grad_help_temp = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *weight_grad_help = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *weight_grad_inp = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *weight_grad_inp_temp = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *weight_grad_inp_old = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *output_value = (float *)calloc(mini_batch_len * output_num, sizeof(float));
    float *weight_trans = (float *)calloc(all_neighbour_num, sizeof(float));

    unsigned long long int grid_rows = (mini_batch_len + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X; // mini_batch_len
    unsigned long long int grid_cols = (all_input_num + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;  // neuron_num
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    float *weight_grad_inp_g, *weight_grad_inp_temp_g, *weight_grad_inp_old_g, *weight_grad_help_g, *datas_g,
        *weight_trans_g, *input_value_g,
        *input_value_old_g, *input_value_orig_g, *neuron_value_g,
        *weight_grad_help_temp_g;

    cudaMalloc((void **)&weight_grad_inp_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&weight_grad_inp_temp_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&weight_grad_inp_old_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&weight_grad_help_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&weight_grad_help_temp_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&datas_g, sizeof(float) * mini_batch_len * (input_num + output_num));
    cudaMalloc((void **)&weight_trans_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&input_value_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&input_value_old_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&input_value_orig_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&neuron_value_g, sizeof(float) * neuron_num * mini_batch_len);

    cudaMemcpy(input_value_g, input_value, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyHostToDevice);
    cudaMemcpy(datas_g, datas, sizeof(float) * mini_batch_len * (input_num + output_num), cudaMemcpyHostToDevice);

    // Transposing the weight matrix for calculating the network
    weight_transpose_gpu<<<(neuron_num + TPB - 1) / TPB, TPB>>>(first_ind_neighbour_g, first_ind_bias_g, first_ind_parent_g,
                                                                neighbour_number_g, bias_number_g, parent_number_g, graph_p_g, graph_p_ind_n_g, weight_g, weight_trans_g, neuron_num);

    // Copying the input data to input_value_g
    copy_input_gpu<<<dimGrid, dimBlock>>>(datas_g, input_value_g, first_ind_bias_g, mini_batch_len, all_input_num, neuron_num, input_num, output_num);

    cudaMemcpy(input_value_old_g, input_value_g, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyDeviceToDevice);
    cudaMemcpy(input_value_orig_g, input_value_g, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyDeviceToDevice);

    cudaMemcpy(neuron_value_g, neuron_value, sizeof(float) * neuron_num * mini_batch_len, cudaMemcpyHostToDevice);

    // Iteration number
    float *error_iter_g;
    cudaMalloc((void **)&error_iter_g, sizeof(float));
    float *error_iter_c = (float *)calloc(1, sizeof(float));

    iter_f = 0;
    error_iter = inf;

    //++++++++++++++++++++++++++//
    //                          //
    // Calculating the network  //
    //                          //
    //++++++++++++++++++++++++++//
    while (error_iter > tol_fixit && iter_f < maxiter_fix)
    {
        iter_f++;

        // Calculating the neuron values
        calc_neuron_mb_gpu<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                  bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len);

        cudaMemcpy(input_value_old_g, input_value_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);
        cudaMemcpy(input_value_g, input_value_orig_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);

        // Main part
        calc_network_mb_gpu<<<dimGrid, dimBlock>>>(datas_g, mini_batch_len, bias_number_g,
                                                   parent_number_g, first_ind_bias_g, first_ind_parent_g, graph_p_g,
                                                   weight_trans_g, neuron_value_g, input_value_g, neuron_num, all_input_num);
        // Adding bias
        add_bias_bcast<<<dimGrid, dimBlock>>>(mini_batch_len, all_input_num, bias_g, input_value_g);

        // Calculating the error and L1-error
        // maxnormDiff<<<(all_input_num * mini_batch_len + TPB - 1) / TPB, TPB>>>(input_value_old_g, input_value_g, all_input_num * mini_batch_len, error_iter_g);
        // l1normdiff<<<(all_input_num * mini_batch_len + TPB - 1) / TPB, TPB>>>(input_value_old_g, input_value_g, all_input_num * mini_batch_len, error_iter_g);
        cudaMemcpy(input_value, input_value_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(input_value_old, input_value_old_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToHost);

        // cudaMemcpy(error_iter_c, error_iter_g, sizeof(float), cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();
        error_iter = calc_diff_vectors(input_value, input_value_old, all_input_num * mini_batch_len);

        // error_iter = error_iter_c[0];
    }

    //++++++++++++++++++++++++++//
    //                          //
    // Calculating the gradient //
    //                          //
    //++++++++++++++++++++++++++//

    calc_grad_help_0_gpu<<<dimGrid, dimBlock>>>(first_ind_neighbour_g, first_ind_bias_g, bias_number_g,
                                                input_value_g, weight_grad_help_temp_g, activation_type_g, neuron_num, all_input_num, mini_batch_len);

    calc_grad_help_gpu<<<dimGrid, dimBlock>>>(first_ind_neighbour_g, first_ind_bias_g, bias_number_g,
                                              input_value_g, weight_grad_help_g, weight_grad_help_temp_g, activation_type_g,
                                              neuron_num, all_input_num, mini_batch_len);

    cudaMemcpy(weight_grad_help, weight_grad_help_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_value, input_value_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_value_old, input_value_old_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(neuron_value, neuron_value_g, sizeof(float) * mini_batch_len * neuron_num, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    for (unsigned long long int data_index = 0; data_index < mini_batch_len; data_index++)
    {
        // Output neurons and targets
        for (unsigned long long int i = 0; i < output_num; i++)
        {
            output_value[data_index * output_num + i] = neuron_value[data_index * neuron_num + neuron_num - output_num + i];
            target_vector[data_index * output_num + i] = datas[data_index * (input_num + output_num) + input_num + i];
        }
        if (strcmp(train_lossfunction_type, "multiclassification_crossentropy") == 0)
        {
            float *output_value_temp = (float *)malloc(output_num * sizeof(float));
            for (unsigned long long int i = 0; i < output_num; i++)
            {
                output_value_temp[i] = output_value[data_index * output_num + i];
            }
            softmax(output_value_temp, output_num);

            for (unsigned long long int i = 0; i < output_num; i++)
            {
                output_value[data_index * output_num + i] = output_value_temp[i];
            }

            free(output_value_temp);
        }

        if (strcmp(train_lossfunction_type, "multiclassification_crossentropy") == 0)
        {

            for (unsigned long long int i = 0; i < output_num; i++)
            {
                unsigned long long int neuron_id = neuron_num - output_num + i;
                unsigned long long int startind = first_ind_bias[neuron_id];
                for (unsigned long long int j = 0; j < bias_number[neuron_id]; j++)
                {
                    weight_grad_inp_old[data_index * all_input_num + startind + j] =
                        (output_value[data_index * output_num + i] - target_vector[data_index * output_num + i]) *
                        weight_grad_help[data_index * all_input_num + startind + j];
                }
            }
        }

        if (strcmp(train_lossfunction_type, "MSE") == 0)
        {

            for (unsigned long long int i = 0; i < output_num; i++)
            {
                unsigned long long int neuron_id = neuron_num - output_num + i;
                unsigned long long int startind = first_ind_bias[neuron_id];
                for (unsigned long long int j = 0; j < bias_number[neuron_id]; j++)
                {
                    weight_grad_inp_old[data_index * all_input_num + startind + j] =
                        (output_value[data_index * output_num + i] - target_vector[data_index * output_num + i]) *
                        weight_grad_help[data_index * all_input_num + startind + j];
                }
            }
        }

        if (strcmp(train_lossfunction_type, "MAE") == 0)
        {

            for (unsigned long long int i = 0; i < output_num; i++)
            {
                unsigned long long int neuron_id = neuron_num - output_num + i;
                unsigned long long int startind = first_ind_bias[neuron_id];
                for (unsigned long long int j = 0; j < bias_number[neuron_id]; j++)
                {
                    float diff_temp = 0.0;
                    diff_temp = output_value[data_index * output_num + i] - target_vector[data_index * output_num + i];
                    if (diff_temp >= 0)
                    {
                        weight_grad_inp_old[data_index * all_input_num + startind + j] = weight_grad_help[data_index * all_input_num + startind + j];
                    }
                    else
                    {
                        weight_grad_inp_old[data_index * all_input_num + startind + j] = -weight_grad_help[data_index * all_input_num + startind + j];
                    }
                }
            }
        }

        if (strcmp(train_lossfunction_type, "multilabeling_crossentropy") == 0)
        {

            for (unsigned long long int i = 0; i < output_num; i++)
            {
                unsigned long long int neuron_id = neuron_num - output_num + i;
                unsigned long long int startind = first_ind_bias[neuron_id];
                for (unsigned long long int j = 0; j < bias_number[neuron_id]; j++)
                {
                    if ((output_value[data_index * output_num + i] > 0.0) && (output_value[data_index * output_num + i] < 1.0))
                    {
                        weight_grad_inp_old[data_index * all_input_num + startind + j] = (output_value[data_index * output_num + i] - target_vector[data_index * output_num + i]) *
                                                                                         weight_grad_help[data_index * all_input_num + startind + j] /
                                                                                         (output_value[data_index * output_num + i] * (1.0 - output_value[data_index * output_num + i]));
                    }
                }
            }
        }

        if (strcmp(train_lossfunction_type, "bce_multilabeling") == 0)
        {

            for (unsigned long long int i = 0; i < output_num; i++)
            {
                unsigned long long int neuron_id = neuron_num - output_num + i;
                unsigned long long int startind = first_ind_bias[neuron_id];
                for (unsigned long long int j = 0; j < bias_number[neuron_id]; j++)
                {
                    weight_grad_inp_old[data_index * all_input_num + startind + j] =
                        act_fun(output_value[data_index * output_num + i], 1) - target_vector[data_index * output_num + i];
                }
            }
        }
    }

    cudaMemcpy(weight_grad_inp_temp_g, weight_grad_inp_old, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_grad_inp_g, weight_grad_inp_old, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_grad_help_g, weight_grad_help, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_grad_inp_old_g, weight_grad_inp_old, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyHostToDevice);

    iter_b = 0;
    error_iter = inf;

    //++++++++++++++++++++++++++//
    //                          //
    //     Back propagation     //
    //                          //
    //++++++++++++++++++++++++++//

    while (iter_b < maxiter_fix && error_iter > tol_fixit)
    {
        iter_b++;

        cudaMemcpy(weight_grad_inp_old_g, weight_grad_inp_temp_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);

        // Main part
        calc_gradient_mb_gpu<<<dimGrid, dimBlock>>>(weight_grad_inp_g, weight_grad_inp_temp_g, weight_grad_inp_old_g,
                                                    weight_grad_help_g, mini_batch_len, neighbour_number_g, bias_number_g,
                                                    first_ind_neighbour_g, first_ind_bias_g,
                                                    graph_n_g, graph_i_g, weight_g, neuron_num, all_input_num);

        // Max abs error and L1-error
        // maxnorm<<<(all_input_num * mini_batch_len + TPB - 1) / TPB, TPB>>>(weight_grad_inp_temp_g, all_input_num * mini_batch_len, error_iter_g);
        // l1norm<<<(all_input_num * mini_batch_len + TPB - 1) / TPB, TPB>>>(weight_grad_inp_temp_g, all_input_num * mini_batch_len, error_iter_g);
        cudaMemcpy(weight_grad_inp_temp, weight_grad_inp_temp_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToHost);

        // cudaMemcpy(error_iter_c, error_iter_g, sizeof(float), cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();
        error_iter = calc_vector_norm(weight_grad_inp_temp, all_input_num * mini_batch_len);

        // error_iter = error_iter_c[0];
    }

    // Calculating the gradients
    calc_gradient_mb_sum_gpu_w<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_grad_inp_g, weight_grad_g, neuron_value_g, first_ind_neighbour_g, first_ind_bias_g, neighbour_number_g, graph_n_g, graph_i_g, neuron_num, all_input_num, all_neighbour_num, mini_batch_len);
    calc_gradient_mb_sum_gpu_b<<<(all_input_num + TPB - 1) / TPB, TPB>>>(weight_grad_inp_g, bias_grad_g, neuron_num, all_input_num, all_neighbour_num, mini_batch_len);
    divide_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_grad_g, all_neighbour_num, mini_batch_len);
    divide_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(bias_grad_g, all_input_num, mini_batch_len);

    // Regularization
    reg_weight_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_g, bias_g, weight_grad_g, bias_grad_g, all_input_num, all_neighbour_num, alpha);

    // Clipping
    if (clipping == 1)
    {
        clipping_weight_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_grad_g, bias_grad_g, all_input_num, all_neighbour_num, clipping_threshold);
        clipping_bias_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(weight_grad_g, bias_grad_g, all_input_num, all_neighbour_num, clipping_threshold);
    }

    cudaThreadSynchronize();

    // Calculate the error
    error_mini_batch = calc_error(neuron_value, target_vector, mini_batch_len);

    cudaFree(weight_grad_inp_g);
    cudaFree(weight_grad_inp_temp_g);
    cudaFree(weight_grad_inp_old_g);
    cudaFree(weight_grad_help_g);
    cudaFree(weight_grad_help_temp_g);
    cudaFree(datas_g);
    cudaFree(weight_trans_g);
    cudaFree(input_value_g);
    cudaFree(input_value_old_g);
    cudaFree(input_value_orig_g);
    cudaFree(neuron_value_g);

    free(input_value);
    free(input_value_old);
    free(input_value_orig);
    free(neuron_value);
    free(target_vector);
    free(weight_grad_help_temp);
    free(weight_grad_help);
    free(weight_grad_inp);
    free(weight_grad_inp_temp);
    free(weight_grad_inp_old);
    free(output_value);
    free(weight_trans);
    free(error_iter_c);

    cudaFree(error_iter_g);

    error_mini_batch /= mini_batch_len;
    *iter_forward = iter_f;
    *iter_backward = iter_b;

    return error_mini_batch;
}

float calc_gradient_mini_batch_ff(float *datas, unsigned long long int mini_batch_len,
                                  unsigned long long int *neighbour_number_g, unsigned long long int *neighbour_number, unsigned long long int *bias_number_g, unsigned long long int *bias_number, unsigned long long int *parent_number_g,
                                  unsigned long long int *first_ind_neighbour_g, unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias_g, unsigned long long int *first_ind_bias, unsigned long long int *first_ind_parent_g,
                                  unsigned long long int *activation_type_g, unsigned long long int *activation_type, unsigned long long int *graph_n_g, unsigned long long int *graph_n, unsigned long long int *graph_i_g, unsigned long long int *graph_i, unsigned long long int *graph_p_g,
                                  unsigned long long int *graph_p_ind_n_g,
                                  float *weight_g, float *bias_g,
                                  float *weight_grad_g, float *bias_grad_g,
                                  float *iter_forward, float *iter_backward,
                                  unsigned long long int dist_max,
                                  unsigned long long int *dist_g, unsigned long long int *dist,
                                  unsigned long long int *dist_input_g, unsigned long long int *dist_input)
{
    // Calculating the gradient on a mini-batch

    // Definitions
    float error_mini_batch = 0.0;
    float iter_forward_temp = 0.0;
    float iter_backward_temp = 0.0;
    unsigned long long int nthreads;

    // Reset gradients
    set_zero_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_grad_g, all_neighbour_num);
    set_zero_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(bias_grad_g, all_input_num);

    //++++++++++++++++++++++++++//
    //                          //
    // Loop over the mini-batch //
    //                          //
    //++++++++++++++++++++++++++//

    unsigned long long int iter_f, iter_b;

    float *input_value = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *neuron_value = (float *)calloc(mini_batch_len * neuron_num, sizeof(float));
    float *target_vector = (float *)calloc(mini_batch_len * output_num, sizeof(float));
    float *weight_grad_help = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *weight_grad_help_temp = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *weight_grad_inp = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *output_value = (float *)calloc(mini_batch_len * output_num, sizeof(float));
    float *weight_trans = (float *)calloc(all_neighbour_num, sizeof(float));

    unsigned long long int grid_rows = (mini_batch_len + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X; // mini_batch_len
    unsigned long long int grid_cols = (all_input_num + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;  // neuron_num
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    float *weight_grad_inp_g, *weight_grad_inp_temp_g, *weight_grad_inp_old_g, *weight_grad_help_g, *datas_g,
        *weight_trans_g, *input_value_g,
        *neuron_value_g,
        *weight_grad_help_temp_g;

    cudaMalloc((void **)&weight_grad_inp_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&weight_grad_inp_old_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&weight_grad_help_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&weight_grad_help_temp_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&datas_g, sizeof(float) * mini_batch_len * (input_num + output_num));
    cudaMalloc((void **)&weight_trans_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&input_value_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&neuron_value_g, sizeof(float) * neuron_num * mini_batch_len);

    cudaMemcpy(input_value_g, input_value, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyHostToDevice);
    cudaMemcpy(datas_g, datas, sizeof(float) * mini_batch_len * (input_num + output_num), cudaMemcpyHostToDevice);

    // Transposing the weight matrix for calculating the network
    weight_transpose_gpu<<<(neuron_num + TPB - 1) / TPB, TPB>>>(first_ind_neighbour_g, first_ind_bias_g, first_ind_parent_g,
                                                                neighbour_number_g, bias_number_g, parent_number_g, graph_p_g, graph_p_ind_n_g, weight_g, weight_trans_g, neuron_num);

    // Copying the input data to input_value_g
    copy_input_gpu<<<dimGrid, dimBlock>>>(datas_g, input_value_g, first_ind_bias_g, mini_batch_len, all_input_num, neuron_num, input_num, output_num);

    // Adding bias
    add_bias_bcast<<<dimGrid, dimBlock>>>(mini_batch_len, all_input_num, bias_g, input_value_g);

    cudaMemcpy(neuron_value_g, neuron_value, sizeof(float) * neuron_num * mini_batch_len, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_grad_help_temp_g, weight_grad_help_temp, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyHostToDevice);

    iter_f = 0;

    //++++++++++++++++++++++++++//
    //                          //
    // Calculating the network  //
    //                          //
    //++++++++++++++++++++++++++//

    // The main loop
    unsigned long long int iter_fix = 0;

    for (unsigned long long int layer_id = 0; layer_id < dist_max; layer_id++) // Here we need `<=` because the indexing of the layers
    {
        iter_f++;
        calc_neuron_mb_gpu_ff<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                     bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len,
                                                     layer_id, dist_g);

        calc_network_mb_gpu_ff<<<dimGrid, dimBlock>>>(datas_g, mini_batch_len, bias_number_g,
                                                      parent_number_g, first_ind_bias_g, first_ind_parent_g, graph_p_g,
                                                      weight_trans_g, neuron_value_g, input_value_g, neuron_num, all_input_num, layer_id + 1, dist_input_g);
    }
    calc_neuron_mb_gpu_ff<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                 bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len,
                                                 dist_max, dist_g);

    //++++++++++++++++++++++++++//
    //                          //
    // Calculating the gradient //
    //                          //
    //++++++++++++++++++++++++++//

    calc_grad_help_0_gpu<<<dimGrid, dimBlock>>>(first_ind_neighbour_g, first_ind_bias_g, bias_number_g,
                                                input_value_g, weight_grad_help_temp_g, activation_type_g, neuron_num, all_input_num, mini_batch_len);

    calc_grad_help_gpu<<<dimGrid, dimBlock>>>(first_ind_neighbour_g, first_ind_bias_g, bias_number_g,
                                              input_value_g, weight_grad_help_g, weight_grad_help_temp_g, activation_type_g,
                                              neuron_num, all_input_num, mini_batch_len);

    cudaMemcpy(weight_grad_help, weight_grad_help_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(neuron_value, neuron_value_g, sizeof(float) * mini_batch_len * neuron_num, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    for (unsigned long long int data_index = 0; data_index < mini_batch_len; data_index++)
    {
        // Output neurons and targets
        for (unsigned long long int i = 0; i < output_num; i++)
        {
            output_value[data_index * output_num + i] = neuron_value[data_index * neuron_num + neuron_num - output_num + i];
            target_vector[data_index * output_num + i] = datas[data_index * (input_num + output_num) + input_num + i];
        }
        if (strcmp(train_lossfunction_type, "multiclassification_crossentropy") == 0)
        {
            float *output_value_temp = (float *)malloc(output_num * sizeof(float));
            for (unsigned long long int i = 0; i < output_num; i++)
            {
                output_value_temp[i] = output_value[data_index * output_num + i];
            }
            softmax(output_value_temp, output_num);

            for (unsigned long long int i = 0; i < output_num; i++)
            {
                output_value[data_index * output_num + i] = output_value_temp[i];
            }

            free(output_value_temp);
        }

        if (strcmp(train_lossfunction_type, "multiclassification_crossentropy") == 0)
        {

            for (unsigned long long int i = 0; i < output_num; i++)
            {
                unsigned long long int neuron_id = neuron_num - output_num + i;
                unsigned long long int startind = first_ind_bias[neuron_id];
                for (unsigned long long int j = 0; j < bias_number[neuron_id]; j++)
                {
                    weight_grad_inp[data_index * all_input_num + startind + j] =
                        (output_value[data_index * output_num + i] - target_vector[data_index * output_num + i]) *
                        weight_grad_help[data_index * all_input_num + startind + j];
                }
            }
        }

        if (strcmp(train_lossfunction_type, "MSE") == 0)
        {

            for (unsigned long long int i = 0; i < output_num; i++)
            {
                unsigned long long int neuron_id = neuron_num - output_num + i;
                unsigned long long int startind = first_ind_bias[neuron_id];
                for (unsigned long long int j = 0; j < bias_number[neuron_id]; j++)
                {
                    weight_grad_inp[data_index * all_input_num + startind + j] =
                        (output_value[data_index * output_num + i] - target_vector[data_index * output_num + i]) *
                        weight_grad_help[data_index * all_input_num + startind + j];
                }
            }
        }

        if (strcmp(train_lossfunction_type, "MAE") == 0)
        {

            for (unsigned long long int i = 0; i < output_num; i++)
            {
                unsigned long long int neuron_id = neuron_num - output_num + i;
                unsigned long long int startind = first_ind_bias[neuron_id];
                for (unsigned long long int j = 0; j < bias_number[neuron_id]; j++)
                {
                    float diff_temp = 0.0;
                    diff_temp = output_value[data_index * output_num + i] - target_vector[data_index * output_num + i];
                    if (diff_temp >= 0)
                    {
                        weight_grad_inp[data_index * all_input_num + startind + j] = weight_grad_help[data_index * all_input_num + startind + j];
                    }
                    else
                    {
                        weight_grad_inp[data_index * all_input_num + startind + j] = -weight_grad_help[data_index * all_input_num + startind + j];
                    }
                }
            }
        }

        if (strcmp(train_lossfunction_type, "multilabeling_crossentropy") == 0)
        {

            for (unsigned long long int i = 0; i < output_num; i++)
            {
                unsigned long long int neuron_id = neuron_num - output_num + i;
                unsigned long long int startind = first_ind_bias[neuron_id];
                for (unsigned long long int j = 0; j < bias_number[neuron_id]; j++)
                {
                    if ((output_value[data_index * output_num + i] > 0.0) && (output_value[data_index * output_num + i] < 1.0))
                    {
                        weight_grad_inp[data_index * all_input_num + startind + j] = (output_value[data_index * output_num + i] - target_vector[data_index * output_num + i]) *
                                                                                     weight_grad_help[data_index * all_input_num + startind + j] /
                                                                                     (output_value[data_index * output_num + i] * (1.0 - output_value[data_index * output_num + i]));
                    }
                }
            }
        }

        if (strcmp(train_lossfunction_type, "bce_multilabeling") == 0)
        {

            for (unsigned long long int i = 0; i < output_num; i++)
            {
                unsigned long long int neuron_id = neuron_num - output_num + i;
                unsigned long long int startind = first_ind_bias[neuron_id];
                for (unsigned long long int j = 0; j < bias_number[neuron_id]; j++)
                {
                    weight_grad_inp[data_index * all_input_num + startind + j] =
                        act_fun(output_value[data_index * output_num + i], 1) - target_vector[data_index * output_num + i];
                }
            }
        }
    }

    cudaMemcpy(weight_grad_inp_g, weight_grad_inp, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_grad_help_g, weight_grad_help, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyHostToDevice);

    iter_b = 0;

    //++++++++++++++++++++++++++//
    //                          //
    //     Back propagation     //
    //                          //
    //++++++++++++++++++++++++++//

    for (unsigned long long int layer_id = dist_max - 1; layer_id > 0; layer_id--)
    {
        iter_b++;

        cudaMemcpy(weight_grad_inp_old_g, weight_grad_inp_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);

        calc_gradient_mb_gpu_ff<<<dimGrid, dimBlock>>>(weight_grad_inp_g, weight_grad_inp_old_g,
                                                       weight_grad_help_g, mini_batch_len, neighbour_number_g, bias_number_g,
                                                       first_ind_neighbour_g, first_ind_bias_g,
                                                       graph_n_g, graph_i_g, weight_g, neuron_num, all_input_num,
                                                       layer_id, dist_g);
    }
    cudaMemcpy(weight_grad_inp_old_g, weight_grad_inp_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);
    calc_gradient_mb_gpu_ff<<<dimGrid, dimBlock>>>(weight_grad_inp_g, weight_grad_inp_old_g,
                                                   weight_grad_help_g, mini_batch_len, neighbour_number_g, bias_number_g,
                                                   first_ind_neighbour_g, first_ind_bias_g,
                                                   graph_n_g, graph_i_g, weight_g, neuron_num, all_input_num,
                                                   0, dist_g);

    // Calculating the gradients
    calc_gradient_mb_sum_gpu_w<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_grad_inp_g, weight_grad_g, neuron_value_g, first_ind_neighbour_g, first_ind_bias_g, neighbour_number_g, graph_n_g, graph_i_g, neuron_num, all_input_num, all_neighbour_num, mini_batch_len);
    calc_gradient_mb_sum_gpu_b<<<(all_input_num + TPB - 1) / TPB, TPB>>>(weight_grad_inp_g, bias_grad_g, neuron_num, all_input_num, all_neighbour_num, mini_batch_len);
    divide_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_grad_g, all_neighbour_num, mini_batch_len);
    divide_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(bias_grad_g, all_input_num, mini_batch_len);

    // Regularization
    reg_weight_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_g, bias_g, weight_grad_g, bias_grad_g, all_input_num, all_neighbour_num, alpha);

    // Clipping
    if (clipping == 1)
    {
        clipping_weight_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_grad_g, bias_grad_g, all_input_num, all_neighbour_num, clipping_threshold);
        clipping_bias_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(weight_grad_g, bias_grad_g, all_input_num, all_neighbour_num, clipping_threshold);
    }

    cudaThreadSynchronize();

    // Calculate the error
    error_mini_batch = calc_error(neuron_value, target_vector, mini_batch_len);

    cudaFree(weight_grad_inp_g);
    cudaFree(weight_grad_inp_old_g);
    cudaFree(weight_grad_help_g);
    cudaFree(weight_grad_help_temp_g);
    cudaFree(datas_g);
    cudaFree(weight_trans_g);
    cudaFree(input_value_g);
    cudaFree(neuron_value_g);

    free(input_value);
    free(neuron_value);
    free(target_vector);
    free(weight_grad_help);
    free(weight_grad_help_temp);
    free(weight_grad_inp);
    free(output_value);
    free(weight_trans);

    error_mini_batch /= mini_batch_len;
    *iter_forward = iter_f;
    *iter_backward = iter_b;

    return error_mini_batch;
}

float calc_network_mini_batch(float *datas, unsigned long long int mini_batch_len,
                              unsigned long long int *neighbour_number_g, unsigned long long int *neighbour_number, unsigned long long int *bias_number_g, unsigned long long int *bias_number, unsigned long long int *parent_number_g,
                              unsigned long long int *first_ind_neighbour_g, unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias_g, unsigned long long int *first_ind_bias, unsigned long long int *first_ind_parent_g,
                              unsigned long long int *activation_type_g, unsigned long long int *activation_type, unsigned long long int *graph_n_g, unsigned long long int *graph_n, unsigned long long int *graph_i_g, unsigned long long int *graph_i, unsigned long long int *graph_p_g,
                              unsigned long long int *graph_p_ind_n_g,
                              float *weight_trans_g, float *bias_g,
                              float *iter_forward)
{
    // Definitions
    float error_mini_batch = 0.0;
    float iter_forward_temp = 0.0;

    unsigned long long int nthreads;

    //++++++++++++++++++++++++++//
    //                          //
    // Loop over the mini-batch //
    //                          //
    //++++++++++++++++++++++++++//

    // Loop over the elements on the elements of the mini-batch

    float error_temp;
    unsigned long long int iter_f;
    float error_iter;

    float *input_value = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *neuron_value = (float *)calloc(mini_batch_len * neuron_num, sizeof(float));
    float *target_vector = (float *)calloc(mini_batch_len * output_num, sizeof(float));
    float *input_value_old = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *input_value_orig = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *output_value = (float *)calloc(mini_batch_len * output_num, sizeof(float));

    float *datas_g,
        *input_value_g,
        *input_value_old_g, *input_value_orig_g, *neuron_value_g;

    cudaMalloc((void **)&datas_g, sizeof(float) * mini_batch_len * (input_num + output_num));
    cudaMalloc((void **)&input_value_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&input_value_old_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&input_value_orig_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&neuron_value_g, sizeof(float) * neuron_num * mini_batch_len);

    cudaMemcpy(datas_g, datas, sizeof(float) * mini_batch_len * (input_num + output_num), cudaMemcpyHostToDevice);
    cudaMemcpy(input_value_g, input_value, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyHostToDevice);

    unsigned long long int grid_rows = (mini_batch_len + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X; // mini_batch_len
    unsigned long long int grid_cols = (all_input_num + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;  // neuron_num
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    // Copying the input data to input_value_g
    copy_input_gpu<<<dimGrid, dimBlock>>>(datas_g, input_value_g, first_ind_bias_g, mini_batch_len, all_input_num, neuron_num, input_num, output_num);

    cudaMemcpy(input_value_old_g, input_value_g, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyDeviceToDevice);
    cudaMemcpy(input_value_orig_g, input_value_g, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyDeviceToDevice);
    cudaMemcpy(neuron_value_g, neuron_value, sizeof(float) * neuron_num * mini_batch_len, cudaMemcpyHostToDevice);

    float *error_iter_g;
    unsigned long long int *maxid_g;

    cudaMalloc((void **)&error_iter_g, sizeof(float));

    float *error_iter_c = (float *)calloc(1, sizeof(float));

    iter_f = 0;
    error_iter = inf;

    //++++++++++++++++++++++++++//
    //                          //
    // Calculating the network  //
    //                          //
    //++++++++++++++++++++++++++//
    while (error_iter > tol_fixit && iter_f < maxiter_fix)
    {
        iter_f++;

        // Calculating the neuron values
        calc_neuron_mb_gpu<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                  bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len);

        cudaMemcpy(input_value_old_g, input_value_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);
        cudaMemcpy(input_value_g, input_value_orig_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);

        // Main part
        calc_network_mb_gpu<<<dimGrid, dimBlock>>>(datas_g, mini_batch_len, bias_number_g,
                                                   parent_number_g, first_ind_bias_g, first_ind_parent_g, graph_p_g,
                                                   weight_trans_g, neuron_value_g, input_value_g, neuron_num, all_input_num);
        // Adding bias
        add_bias_bcast<<<dimGrid, dimBlock>>>(mini_batch_len, all_input_num, bias_g, input_value_g);

        // Calculating the error
        // maxnormDiff<<<(all_input_num * mini_batch_len + TPB - 1) / TPB, TPB>>>(input_value_old_g, input_value_g, all_input_num * mini_batch_len, error_iter_g);
        // l1normdiff<<<(all_input_num * mini_batch_len + TPB - 1) / TPB, TPB>>>(input_value_old_g, input_value_g, all_input_num * mini_batch_len, error_iter_g);
        cudaMemcpy(input_value, input_value_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(input_value_old, input_value_old_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToHost);

        // cudaMemcpy(error_iter_c, error_iter_g, sizeof(float), cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();
        error_iter = calc_diff_vectors(input_value, input_value_old, all_input_num * mini_batch_len);
    }

    for (unsigned long long int data_index = 0; data_index < mini_batch_len; data_index++)
    {
        // Targets
        for (unsigned long long int i = 0; i < output_num; i++)
        {
            target_vector[data_index * output_num + i] = datas[data_index * (input_num + output_num) + input_num + i];
        }
    }

    cudaMemcpy(neuron_value, neuron_value_g, sizeof(float) * mini_batch_len * neuron_num, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    // Calculate the error
    error_mini_batch = calc_error(neuron_value, target_vector, mini_batch_len);

    cudaFree(datas_g);
    cudaFree(input_value_g);
    cudaFree(input_value_old_g);
    cudaFree(input_value_orig_g);
    cudaFree(neuron_value_g);

    free(input_value);
    free(input_value_old);
    free(input_value_orig);
    free(neuron_value);
    free(target_vector);
    free(output_value);
    free(error_iter_c);

    cudaFree(error_iter_g);

    error_mini_batch /= mini_batch_len;
    *iter_forward = iter_f;

    return error_mini_batch;
}

float calc_network_mini_batch_ff(float *datas, unsigned long long int mini_batch_len,
                                 unsigned long long int *neighbour_number_g, unsigned long long int *neighbour_number, unsigned long long int *bias_number_g, unsigned long long int *bias_number, unsigned long long int *parent_number_g,
                                 unsigned long long int *first_ind_neighbour_g, unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias_g, unsigned long long int *first_ind_bias, unsigned long long int *first_ind_parent_g,
                                 unsigned long long int *activation_type_g, unsigned long long int *activation_type, unsigned long long int *graph_n_g, unsigned long long int *graph_n, unsigned long long int *graph_i_g, unsigned long long int *graph_i, unsigned long long int *graph_p_g,
                                 unsigned long long int *graph_p_ind_n_g,
                                 float *weight_trans_g, float *bias_g,
                                 float *iter_forward,
                                 unsigned long long int dist_max,
                                 unsigned long long int *dist_g, unsigned long long int *dist,
                                 unsigned long long int *dist_input_g, unsigned long long int *dist_input)
{

    unsigned long long int iter_f = 0;
    float error_iter = inf, error_mini_batch = 0.0;

    float *neuron_value = (float *)calloc(mini_batch_len * neuron_num, sizeof(float));
    float *target_vector = (float *)calloc(mini_batch_len * output_num, sizeof(float));
    float *input_value = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *datas_g, *neuron_value_g, *input_value_g;

    cudaMalloc((void **)&datas_g, sizeof(float) * mini_batch_len * (input_num + output_num));
    cudaMalloc((void **)&neuron_value_g, sizeof(float) * neuron_num * mini_batch_len);
    cudaMalloc((void **)&input_value_g, sizeof(float) * all_input_num * mini_batch_len);

    cudaMemcpy(datas_g, datas, sizeof(float) * mini_batch_len * (input_num + output_num), cudaMemcpyHostToDevice);
    cudaMemcpy(input_value_g, input_value, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyHostToDevice);
    cudaMemcpy(neuron_value_g, neuron_value, sizeof(float) * neuron_num * mini_batch_len, cudaMemcpyHostToDevice);

    unsigned long long int grid_rows = (mini_batch_len + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X; // mini_batch_len
    unsigned long long int grid_cols = (all_input_num + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;  // neuron_num
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    // Copying the input data to input_value_g
    copy_input_gpu<<<dimGrid, dimBlock>>>(datas_g, input_value_g, first_ind_bias_g, mini_batch_len, all_input_num, neuron_num, input_num, output_num);
    // Adding bias
    add_bias_bcast<<<dimGrid, dimBlock>>>(mini_batch_len, all_input_num, bias_g, input_value_g);

    // The main loop
    unsigned long long int iter_fix = 0;

    for (unsigned long long int layer_id = 0; layer_id < dist_max; layer_id++) // Here we need `<=` because the indexing of the layers
    {
        iter_fix++;
        calc_neuron_mb_gpu_ff<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                     bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len,
                                                     layer_id, dist_g);

        calc_network_mb_gpu_ff<<<dimGrid, dimBlock>>>(datas_g, mini_batch_len, bias_number_g,
                                                      parent_number_g, first_ind_bias_g, first_ind_parent_g, graph_p_g,
                                                      weight_trans_g, neuron_value_g, input_value_g, neuron_num, all_input_num, layer_id + 1, dist_input_g);
    }
    calc_neuron_mb_gpu_ff<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                 bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len,
                                                 dist_max, dist_g);

    // Targets
    for (unsigned long long int data_index = 0; data_index < mini_batch_len; data_index++)
    {
        for (unsigned long long int i = 0; i < output_num; i++)
        {
            target_vector[data_index * output_num + i] = datas[data_index * (input_num + output_num) + input_num + i];
        }
    }

    cudaMemcpy(neuron_value, neuron_value_g, sizeof(float) * mini_batch_len * neuron_num, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    // Calculate the error
    error_mini_batch = calc_error(neuron_value, target_vector, mini_batch_len);

    cudaFree(datas_g);
    cudaFree(neuron_value_g);
    cudaFree(input_value_g);

    free(neuron_value);
    free(target_vector);
    free(input_value);

    error_mini_batch /= mini_batch_len;
    *iter_forward = iter_f;

    return error_mini_batch;
}

void make_predictions_ff(float *datas, unsigned long long int mini_batch_len,
                         unsigned long long int *neighbour_number_g, unsigned long long int *neighbour_number, unsigned long long int *bias_number_g, unsigned long long int *bias_number, unsigned long long int *parent_number_g,
                         unsigned long long int *first_ind_neighbour_g, unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias_g, unsigned long long int *first_ind_bias, unsigned long long int *first_ind_parent_g,
                         unsigned long long int *activation_type_g, unsigned long long int *activation_type, unsigned long long int *graph_n_g, unsigned long long int *graph_n, unsigned long long int *graph_i_g, unsigned long long int *graph_i, unsigned long long int *graph_p_g,
                         unsigned long long int *graph_p_ind_n_g,
                         float *weight_trans_g, float *bias_g,
                         unsigned long long int dist_max,
                         unsigned long long int *dist_g, unsigned long long int *dist,
                         unsigned long long int *dist_input_g, unsigned long long int *dist_input,
                         float **predictions_mini_batch)
{
    unsigned long long int iter_f = 0;

    for (unsigned long long int i = 0; i < mini_batch_len; i++)
    {
        for (unsigned long long int j = 0; j < output_num; j++)
        {
            predictions_mini_batch[i][j] = 0.0;
        }
    }

    float *neuron_value = (float *)calloc(mini_batch_len * neuron_num, sizeof(float));
    float *target_vector = (float *)calloc(mini_batch_len * output_num, sizeof(float));
    float *input_value = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *output_value = (float *)calloc(output_num, sizeof(float));

    float *datas_g, *neuron_value_g, *input_value_g;

    cudaMalloc((void **)&datas_g, sizeof(float) * mini_batch_len * (input_num + output_num));
    cudaMalloc((void **)&neuron_value_g, sizeof(float) * neuron_num * mini_batch_len);
    cudaMalloc((void **)&input_value_g, sizeof(float) * all_input_num * mini_batch_len);

    cudaMemcpy(datas_g, datas, sizeof(float) * mini_batch_len * (input_num + output_num), cudaMemcpyHostToDevice);
    cudaMemcpy(input_value_g, input_value, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyHostToDevice);
    cudaMemcpy(neuron_value_g, neuron_value, sizeof(float) * neuron_num * mini_batch_len, cudaMemcpyHostToDevice);

    unsigned long long int grid_rows = (mini_batch_len + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X; // mini_batch_len
    unsigned long long int grid_cols = (all_input_num + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;  // neuron_num
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    // Copying the input data to input_value_g
    copy_input_gpu<<<dimGrid, dimBlock>>>(datas_g, input_value_g, first_ind_bias_g, mini_batch_len, all_input_num, neuron_num, input_num, output_num);
    // Adding bias
    add_bias_bcast<<<dimGrid, dimBlock>>>(mini_batch_len, all_input_num, bias_g, input_value_g);

    // The main loop
    unsigned long long int iter_fix = 0;

    for (unsigned long long int layer_id = 0; layer_id < dist_max; layer_id++) // Here we need `<=` because the indexing of the layers
    {
        iter_fix++;
        calc_neuron_mb_gpu_ff<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                     bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len,
                                                     layer_id, dist_g);

        calc_network_mb_gpu_ff<<<dimGrid, dimBlock>>>(datas_g, mini_batch_len, bias_number_g,
                                                      parent_number_g, first_ind_bias_g, first_ind_parent_g, graph_p_g,
                                                      weight_trans_g, neuron_value_g, input_value_g, neuron_num, all_input_num, layer_id + 1, dist_input_g);
    }
    calc_neuron_mb_gpu_ff<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                 bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len,
                                                 dist_max, dist_g);

    // Targets
    for (unsigned long long int data_index = 0; data_index < mini_batch_len; data_index++)
    {
        for (unsigned long long int i = 0; i < output_num; i++)
        {
            target_vector[data_index * output_num + i] = datas[data_index * (input_num + output_num) + input_num + i];
        }
    }

    cudaMemcpy(neuron_value, neuron_value_g, sizeof(float) * mini_batch_len * neuron_num, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    // Copying the predictions to predictions_mini_batch
    for (unsigned long long int data_index = 0; data_index < mini_batch_len; data_index++)
    {
        // Calculate the output
        for (unsigned long long int i = 0; i < output_num; i++)
        {
            unsigned long long int neuron_id = neuron_num - output_num + i;
            output_value[i] = neuron_value[data_index * neuron_num + neuron_id];
        }

        if (strcmp(train_lossfunction_type, "multiclassification_crossentropy") == 0)
        {

            softmax(output_value, output_num);
        }

        if (strcmp(train_lossfunction_type, "bce_multilabeling") == 0)
        {
            for (unsigned long long int i = 0; i < output_num; i++)
            {

                output_value[i] = act_fun(output_value[i], 1);
            }
        }

        for (unsigned long long int j = 0; j < output_num; j++)
        {
            predictions_mini_batch[data_index][j] = output_value[j];
        }
    }

    cudaFree(datas_g);
    cudaFree(neuron_value_g);
    cudaFree(input_value_g);

    free(neuron_value);
    free(target_vector);
    free(input_value);
    free(output_value);
}

void make_predictions(float *datas, unsigned long long int mini_batch_len,
                      unsigned long long int *neighbour_number_g, unsigned long long int *neighbour_number, unsigned long long int *bias_number_g, unsigned long long int *bias_number, unsigned long long int *parent_number_g,
                      unsigned long long int *first_ind_neighbour_g, unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias_g, unsigned long long int *first_ind_bias, unsigned long long int *first_ind_parent_g,
                      unsigned long long int *activation_type_g, unsigned long long int *activation_type, unsigned long long int *graph_n_g, unsigned long long int *graph_n, unsigned long long int *graph_i_g, unsigned long long int *graph_i, unsigned long long int *graph_p_g,
                      unsigned long long int *graph_p_ind_n_g,
                      float *weight_trans_g, float *bias_g,
                      float **predictions_mini_batch)
{
    //
    // Creating predictions
    //

    for (unsigned long long int i = 0; i < mini_batch_len; i++)
    {
        for (unsigned long long int j = 0; j < output_num; j++)
        {
            predictions_mini_batch[i][j] = 0.0;
        }
    }

    float iter_forward_temp = 0.0;
    unsigned long long int nthreads;

    //++++++++++++++++++++++++++//
    //                          //
    // Loop over the mini-batch //
    //                          //
    //++++++++++++++++++++++++++//

    float error_temp;
    unsigned long long int iter_f;
    float error_iter;

    float *input_value = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *neuron_value = (float *)calloc(mini_batch_len * neuron_num, sizeof(float));
    float *target_vector = (float *)calloc(mini_batch_len * output_num, sizeof(float));
    float *input_value_old = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *input_value_orig = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *output_value = (float *)calloc(output_num, sizeof(float));
    float *datas_g, *input_value_g,
        *input_value_old_g, *input_value_orig_g, *neuron_value_g, *neuron_value_temp_g;

    cudaMalloc((void **)&datas_g, sizeof(float) * mini_batch_len * (input_num + output_num));
    cudaMalloc((void **)&input_value_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&input_value_old_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&input_value_orig_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&neuron_value_g, sizeof(float) * neuron_num * mini_batch_len);

    cudaMemcpy(datas_g, datas, sizeof(float) * mini_batch_len * (input_num + output_num), cudaMemcpyHostToDevice);

    unsigned long long int grid_rows = (mini_batch_len + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X; // mini_batch_len
    unsigned long long int grid_cols = (all_input_num + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;  // neuron_num
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    cudaMemcpy(input_value_g, input_value, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyHostToDevice);
    // Copying the input data to input_value_g
    copy_input_gpu<<<dimGrid, dimBlock>>>(datas_g, input_value_g, first_ind_bias_g, mini_batch_len, all_input_num, neuron_num, input_num, output_num);

    cudaMemcpy(input_value_old_g, input_value_g, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyDeviceToDevice);
    cudaMemcpy(input_value_orig_g, input_value_g, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyDeviceToDevice);
    cudaMemcpy(neuron_value_g, neuron_value, sizeof(float) * neuron_num * mini_batch_len, cudaMemcpyHostToDevice);

    float *error_iter_g;

    cudaMalloc((void **)&error_iter_g, sizeof(float));

    float *error_iter_c = (float *)calloc(1, sizeof(float));

    iter_f = 0;
    error_iter = inf;

    //++++++++++++++++++++++++++//
    //                          //
    // Calculating the network  //
    //                          //
    //++++++++++++++++++++++++++//
    while (error_iter > tol_fixit && iter_f < maxiter_fix)
    {
        iter_f++;

        // Calculating the neuron values
        calc_neuron_mb_gpu<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                  bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len);

        cudaMemcpy(input_value_old_g, input_value_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);
        cudaMemcpy(input_value_g, input_value_orig_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);

        // Main part
        calc_network_mb_gpu<<<dimGrid, dimBlock>>>(datas_g, mini_batch_len, bias_number_g,
                                                   parent_number_g, first_ind_bias_g, first_ind_parent_g, graph_p_g,
                                                   weight_trans_g, neuron_value_g, input_value_g, neuron_num, all_input_num);
        // Adding bias
        add_bias_bcast<<<dimGrid, dimBlock>>>(mini_batch_len, all_input_num, bias_g, input_value_g);

        // Calculating the error
        // maxnormDiff<<<(all_input_num * mini_batch_len + TPB - 1) / TPB, TPB>>>(input_value_old_g, input_value_g, all_input_num * mini_batch_len, error_iter_g);
        // l1normdiff<<<(all_input_num * mini_batch_len + TPB - 1) / TPB, TPB>>>(input_value_old_g, input_value_g, all_input_num * mini_batch_len, error_iter_g);

        cudaMemcpy(input_value, input_value_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(input_value_old, input_value_old_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToHost);

        // cudaMemcpy(error_iter_c, error_iter_g, sizeof(float), cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();
        error_iter = calc_diff_vectors(input_value, input_value_old, all_input_num * mini_batch_len);
    }
    cudaMemcpy(neuron_value, neuron_value_g, sizeof(float) * mini_batch_len * neuron_num, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    // Copying the predictions to predictions_mini_batch
    for (unsigned long long int data_index = 0; data_index < mini_batch_len; data_index++)
    {
        // Calculate the output

        for (unsigned long long int i = 0; i < output_num; i++)
        {
            unsigned long long int neuron_id = neuron_num - output_num + i;
            output_value[i] = neuron_value[data_index * neuron_num + neuron_id];
        }

        if (strcmp(train_lossfunction_type, "multiclassification_crossentropy") == 0)
        {

            softmax(output_value, output_num);
        }

        if (strcmp(train_lossfunction_type, "bce_multilabeling") == 0)
        {
            for (unsigned long long int i = 0; i < output_num; i++)
            {

                output_value[i] = act_fun(output_value[i], 1);
            }
        }

        for (unsigned long long int j = 0; j < output_num; j++)
        {
            predictions_mini_batch[data_index][j] = output_value[j];
        }
    }

    cudaFree(datas_g);
    cudaFree(input_value_g);
    cudaFree(input_value_old_g);
    cudaFree(input_value_orig_g);
    cudaFree(neuron_value_g);

    free(input_value);
    free(input_value_old);
    free(input_value_orig);
    free(neuron_value);
    free(target_vector);
    free(output_value);
    free(error_iter_c);

    cudaFree(error_iter_g);
}

void save_weight_bias(char filename[100], float *weight, float *bias,
                      unsigned long long int neuron_num, unsigned long long int *neighbour_number, unsigned long long int *bias_number,
                      unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias)
{
    FILE *f = fopen(filename, "w");
    if (f)
    {
        for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            unsigned long long int startind = first_ind_neighbour[neuron_id];
            for (unsigned long long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {

                fprintf(f, "%f ", weight[startind + neighbour_ind]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
        for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            unsigned long long int startind = first_ind_bias[neuron_id];
            for (unsigned long long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {
                fprintf(f, "%f ", bias[startind + bias_ind]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "%f %f %f ", valid_th_best, valid_mean_best, valid_std_best);
        fprintf(f, "\n");
    }
    else
    {
        program_failure("File write error in backup file!");
    }
    fclose(f);
}

void load_weight_bias(char filename[100], float *weight, float *bias,
                      unsigned long long int neuron_num, unsigned long long int *neighbour_number, unsigned long long int *bias_number,
                      unsigned long long int *first_ind_neighbour, unsigned long long int *first_ind_bias)
{
    FILE *f = fopen(filename, "r");
    if (f)
    {
        for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            unsigned long long int startind = first_ind_neighbour[neuron_id];
            for (unsigned long long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                fscanf(f, "%f", &weight[startind + neighbour_ind]);
            }
        }

        for (unsigned long long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            unsigned long long int startind = first_ind_bias[neuron_id];
            for (unsigned long long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {
                fscanf(f, "%f ", &bias[startind + bias_ind]);
            }
        }
        fscanf(f, "%f %f %f ", &valid_th_best, &valid_mean_best, &valid_std_best);
    }
    else
    {
        program_failure("File read error in backup file!");
    }
    fclose(f);
}

float dmax(float a, float b)
{
    /**
     * Returns max(a,b) --- float
     */
    if (a > b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

float calc_vector_norm(float *v1, unsigned long long int row_nums)
{
    //
    // Calculating the maximum norm of a vector
    //
    /*
    float norm = 0.0;
    for (unsigned long long int i = 0; i < row_nums; i++)
    {
        if (fabsf(v1[i]) > norm)
        {
            norm = fabsf(v1[i]);
        }
        // norm = norm > fabs(v1[i]) ? norm : fabs(v1[i]);
    }
    */

    float norm = 0.0;
#pragma omp parallel for reduction(max \
                                   : norm)
    for (unsigned long long int i = 0; i < row_nums; i++)
    {
        norm = norm > fabsf(v1[i]) ? norm : fabsf(v1[i]);
    }
    return norm;
}

float calc_vector_max(float *v1, unsigned long long int row_nums)
{
    //
    // Calculating the maximum of a vector
    //
    /*
    float norm = 0.0;
    for (unsigned long long int i = 0; i < row_nums; i++)
    {
        if (fabsf(v1[i]) > norm)
        {
            norm = fabsf(v1[i]);
        }
        // norm = norm > fabs(v1[i]) ? norm : fabs(v1[i]);
    }
    */

    float norm = v1[0];
#pragma omp parallel for reduction(max \
                                   : norm)
    for (unsigned long long int i = 0; i < row_nums; i++)
    {
        norm = norm > v1[i] ? norm : fabsf(v1[i]);
    }
    return norm;
}

float calc_diff_vectors(float *v1, float *v2, unsigned long long int row_nums)
{
    //
    // Calculating the maximum norm of the difference of two vectors
    //
    float norm = 0.0;
#pragma omp parallel for reduction(max \
                                   : norm)
    for (unsigned long long int i = 0; i < row_nums; i++)
    {
        norm = norm > fabsf(v1[i] - v2[i]) ? norm : fabsf(v1[i] - v2[i]);
    }

    return norm;
}

float trapz(float *x, float *y, unsigned long long int x_size)
{
    //
    // Trapezoidal rule for calculating auc
    //
    float area = 0.0;
    for (unsigned long long int i = 0; i < x_size - 1; i++)
    {
        float dx = x[i + 1] - x[i];
        area += dx * (y[i + 1] + y[i]) * 0.5;
    }
    return area;
}

float calculate_auc(float *fpr, float *tpr, unsigned long long int fpr_size)
{
    //
    // Calculating auc
    //
    float auc = 0.0;

    // sort fpr in ascending order (according to this arrange tpr too)
    for (unsigned long long int i = 0; i < fpr_size - 1; i++)
    {
        for (unsigned long long int j = i + 1; j < fpr_size; j++)
        {
            if (fpr[j] < fpr[i])
            {
                float fpr_temp = fpr[i];
                fpr[i] = fpr[j];
                fpr[j] = fpr_temp;

                float tpr_temp = tpr[i];
                tpr[i] = tpr[j];
                tpr[j] = tpr_temp;
            }
        }
    }

    auc = trapz(fpr, tpr, fpr_size);
    return auc;
}

void calculate_tpr_fpr_bc(unsigned long long int *predictions, unsigned long long int *targets, unsigned long long int length, float *tpr, float *fpr, float *f1score, float *accuracy, float *precision)
{
    //
    // Calculating fpr, tpr (recall), precision, f1-score and accuracy for binary classification
    //
    // calling: calculate_tpr_fpr_bc(predictions, targets, length, &tpr, &fpr, &f1score, &precision, &accuracy);
    //
    //
    unsigned long long int tp = 0, fp = 0, tn = 0, fn = 0;

    for (unsigned long long int i = 0; i < length; i++)
    {
        if (targets[i] == 1 && predictions[i] == 1)
        {
            tp++;
        }
        else if (targets[i] == 0 && predictions[i] == 1)
        {
            fp++;
        }
        else if (targets[i] == 0 && predictions[i] == 0)
        {
            tn++;
        }
        else if (targets[i] == 1 && predictions[i] == 0)
        {
            fn++;
        }
    }

    float recall = (float)(tp) / (float)(tp + fn); // recall
    *fpr = (float)(fp) / (float)(tn + fp);
    *tpr = recall;

    float precision_temp = (float)(tp) / (float)(tp + fp);
    *f1score = 2.0 * precision_temp * recall / (precision_temp + recall);
    *accuracy = ((float)(tp) + (float)(tn)) / ((float)(tp) + (float)(tn) + (float)(fp) + (float)(fn));
    *precision = precision_temp;
}

float calculate_mcc(unsigned long long int *predictions, unsigned long long int *targets, unsigned long long int length)
{
    //
    // Calculating mcc
    //
    // calling: mcc = calculate_mcc(predictions, targets, length);
    //
    //
    unsigned long long int tp = 0, fp = 0, tn = 0, fn = 0;

    for (unsigned long long int i = 0; i < length; i++)
    {
        if (targets[i] == 1 && predictions[i] == 1)
        {
            tp++;
        }
        else if (targets[i] == 0 && predictions[i] == 1)
        {
            fp++;
        }
        else if (targets[i] == 0 && predictions[i] == 0)
        {
            tn++;
        }
        else if (targets[i] == 1 && predictions[i] == 0)
        {
            fn++;
        }
    }
    float mcc = 0.0;
    if (sqrtf(((float)(tp) + (float)(fp)) * ((float)(tp) + (float)(fn)) * ((float)(tn) + (float)(fp)) * ((float)(tn) + (float)(fn))) > 0)
    {
        mcc = ((float)(tp) * (float)(tn) - (float)(fp) * (float)(fn)) / sqrtf(((float)(tp) + (float)(fp)) * ((float)(tp) + (float)(fn)) * ((float)(tn) + (float)(fp)) * ((float)(tn) + (float)(fn)));
    }

    return mcc;
}
