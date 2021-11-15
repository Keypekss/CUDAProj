#include <iostream>
#include <thread>
#include <algorithm>
#include <vector>

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
    return lhs.Position > rhs.Position;
}

void PrintPositions(std::vector<Runner> runner) {
    // array is sorted before printing so that the
    // result is in rank order
    std::sort(std::begin(runner), std::end(runner));

    for (int i = 0; i < NumRunner; ++i) {
        std::cout << "\n";
        std::cout << runner.at(i).ID << ": " << runner.at(i).Position;
    }
    std::cout << "\n\n";
}

int main()
{
    std::vector<Runner> runners; runners.resize(100);
    bool hasFinishedRace = false;
    std::srand(std::time(nullptr)); // use current time as seed for random generator

    // initialize runners
    for (int i = 0; i < NumRunner; ++i) {
        runners[i] = Runner();
        runners[i].ID = i;
    }

    // continue until a runner has finished the race
    while (!hasFinishedRace) {
        // make runners run
        for (int i = 0; i < NumRunner; ++i) {
            runners[i].Position += runners[i].Speed;

            if (runners[i].Position >= Distance)
                hasFinishedRace = true;
        }
        PrintPositions(runners);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}