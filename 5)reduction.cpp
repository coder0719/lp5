#include <iostream>
#include <omp.h>
#include <climits>
using namespace std;

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    int* arr = new int[n];
    cout << "Enter " << n << " elements:\n";
    for (int i = 0; i < n; ++i)
        cin >> arr[i];

    int min_val = INT_MAX;
    int max_val = INT_MIN;
    long long sum = 0;

    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val) reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        min_val = min(min_val, arr[i]);
        max_val = max(max_val, arr[i]);
        sum += arr[i];
    }

    double average = static_cast<double>(sum) / n;

    cout << "\nResults using Parallel Reduction:\n";
    cout << "Minimum: " << min_val << endl;
    cout << "Maximum: " << max_val << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << average << endl;

    delete[] arr;
    return 0;
}
