#include "main_jetson.h"
#include <stdio.h>
#include <stdlib.h>

#include "Operations.h"
#include "QuantOperations.h"
#include "utils.h"
#include "time.h"

#include "sima_data.h"

#define RUNXTIMES 1

clock_t start, end, start_mhsa, end_mhsa;

int main(int argc, char *argv[])
{
    // Initialize the system
    float output_SimA[SIMA_LEN * SIMA_EMBEDDING];
    float sqrtDim_head = sqrtf(SIMA_EMBEDDING/SIMA_HEADS);
    

        // parse number of runs from argv[1], default to RUNXTIMES
    long runs = RUNXTIMES;
    if (argc > 1) {
        char *endptr = NULL;
        long val = strtol(argv[1], &endptr, 10);
        if (endptr != argv[1] && val > 0) runs = val;
        else {
            printf("Invalid runs argument '%s', using default %d\n", argv[1], RUNXTIMES);
        }
    }

    float* time_used_sima = malloc(sizeof(float) * runs * SIMA_COUNT);
    float* time_used_mhsa = malloc(sizeof(float) * runs * SIMA_COUNT);

    // Perform core operations
    for(int a = 0; a < runs; a++){
        for(int i = 0; i < SIMA_COUNT; i++){
            start = clock();
            SimMHAttention(input[i], Wq, Wk, Wv, Wo, Wo_bias, output_SimA, SIMA_HEADS, SIMA_LEN, SIMA_EMBEDDING, SIMA_EMBEDDING, sqrtDim_head);
            end = clock();

            start_mhsa = clock();
            multiHeadAttentionEngine(input[i], Wq, Wk, Wv, Wo, output_SimA, SIMA_HEADS, SIMA_LEN, SIMA_EMBEDDING, SIMA_EMBEDDING, sqrtDim_head);
            end_mhsa = clock();


            // float percmeanE = percMeanError(output_SimA, output[i], SIMA_SAMPLE_SIZE);
            // float percmaxE = percMaxError(output_SimA, output[i], SIMA_SAMPLE_SIZE);

            double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
            double cpu_time_used_mhsa = ((double) (end_mhsa - start_mhsa)) / CLOCKS_PER_SEC;
            printf("Time taken: %f miliseconds\n", cpu_time_used*1000.0);
            printf("Time taken MHSA: %f miliseconds\n", cpu_time_used_mhsa*1000.0);
            // printf("Percentage Mean Error: %f\n", percmeanE);
            // printf("Percentage Max Error: %f\n", percmaxE);

            time_used_sima[a * SIMA_COUNT + i] = cpu_time_used*1000.0;
            time_used_mhsa[a * SIMA_COUNT + i] = cpu_time_used_mhsa*1000.0;
        }
    }
    // Compute average time over runs
    double sum_sima = 0.0;
    double sum_mhsa = 0.0;
    for(int i = 0; i < SIMA_COUNT*runs; i++){
        sum_sima += time_used_sima[i];
        sum_mhsa += time_used_mhsa[i];
    }
    double avg_sima = sum_sima / (SIMA_COUNT * runs);
    double avg_mhsa = sum_mhsa / (SIMA_COUNT * runs);
    printf("Average Time taken SimA: %f miliseconds over %d runs\n", avg_sima, runs);
    printf("Average Time taken MHSA: %f miliseconds over %d runs\n", avg_mhsa, runs);
    
    printf("Speedup SimA vs MHSA: %fx\n", avg_mhsa/avg_sima);
    
    free(time_used_sima);
    free(time_used_mhsa);

    // Finalize and clean up

    return 0;
}