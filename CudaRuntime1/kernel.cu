#include <iostream>
#include <thread>
#include <algorithm>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NumRunner 100
#define Distance 100

struct Runner {
    int ID;
    int Speed;
    int Position;

    Runner() {
        ID = 0;
        Speed = rand() % 5 + 1;
        Position = 0;
    }
};

// comparison operator for Runner struct
inline bool operator <(const Runner& lhs, const Runner& rhs)
{
    return lhs.Position < rhs.Position;
}

void PrintPositions(Runner runners[NumRunner]) {
    // array is sorted before printing so that the
    // result is in rank order
    std::vector<Runner> temp(runners, runners + sizeof(runners) / sizeof(runners[0]));
    std::sort(std::begin(temp), std::end(temp));

    for (int i = 0; i < NumRunner; i++) {
        std::cout << "\n";
        std::cout << runners[i].ID << ": " << runners[i].Position;
    }
    std::cout << "\n\n";
}

__global__ void CalculateDisplacement(Runner runners[], bool* hasFinishedRace) {
   
    int index = threadIdx.x;

    // prevent invoking more threads than number of runners
    if (index < NumRunner) {
        // make runners run        
        runners[index].Position += runners[index].Speed;

        if (runners[index].Position >= Distance) {
            *hasFinishedRace = true;
        }
    }
}

int main()
{
    // allocate memory for runners
    Runner* runners; 
    cudaMallocManaged(&runners, NumRunner * sizeof(Runner));

    // allocate memory for race finish condition
    bool* hasFinishedRace = new bool(false);
    cudaMallocManaged(&hasFinishedRace, sizeof(bool));

    
    std::srand(std::time(nullptr)); // use current time as seed for random generator

    // initialize runners
    for (int i = 0; i < NumRunner; ++i) {
        runners[i] = Runner();
        runners[i].ID = i;
    }

    while (!*hasFinishedRace) {
        CalculateDisplacement <<<1, 128>>> (runners, hasFinishedRace);
        cudaDeviceSynchronize();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        PrintPositions(runners);
    }   

    cudaFree(runners);
    cudaFree(hasFinishedRace);
    
}

