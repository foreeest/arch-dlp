#include <omp.h>
#include <iostream>

int main() {
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        std::cout << "Hello from thread " << thread_id << " of " << total_threads << "\n";
    }
    return 0;
}
