#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

using namespace std;

using ID = uint64_t;

static constexpr double INF = numeric_limits<double>::infinity();
static constexpr double EPS = 1e-9;




// ========================================================
// STATE
// ========================================================

class State {
public:
    ID id;
    vector<double> embedding;

    State() = default;

    State(ID id_, const vector<double>& e)
        : id(id_), embedding(e) {}
};


// ========================================================
// TRANSITION
// ========================================================

class Transition {
public:
    ID id;
    ID from;
    ID to;

    double cost;
    double safety;
    double reliability;

    bool available;

    Transition() = default;

    Transition(
        ID id_,
        ID from_,
        ID to_,
        double cost_,
        double safety_,
        double reliability_,
        bool available_
    )
        : id(id_),
          from(from_),
          to(to_),
          cost(cost_),
          safety(safety_),
          reliability(reliability_),
          available(available_) {}
};


// ========================================================
// PLANNING PROBLEM
// ========================================================

class PlanningProblem {
public:
    ID initialState;
    ID goalState;

    vector<ID> badStates;
    vector<State> states;
    vector<Transition> transitions;
};


// ========================================================
// PLANNING RESULT
// ========================================================

class PlanningResult {
public:
    bool success = false;

    vector<ID> statePath;
    vector<ID> transitionPath;

    double totalCost = INF;

    // Minimum Euclidean distance from a visited state
    // to the nearest bad state.
    double safetyScore = 0.0;

    double cumulativeReliability = 0.0;

    size_t exploredStates = 0;

    double planningTimeMs = 0.0;

    size_t memoryEstimateBytes = 0;
};


// ========================================================
// PLANNER INTERFACE
// ========================================================

class Planner {
public:
    virtual PlanningResult plan(
        const PlanningProblem& problem
    ) = 0;

    virtual ~Planner() = default;
};


// ========================================================
// D* LITE NODE
// ========================================================

struct Key {
    double first;
    double second;

    bool operator<(const Key& other) const {
        if (fabs(first - other.first) > EPS)
            return first < other.first;

        return second < other.second;
    }
};


struct QueueNode {
    ID state;
    Key key;

    bool operator<(const QueueNode& other) const {
        // priority_queue is max heap,
        // so reverse the comparison.
        if (fabs(key.first - other.key.first) > EPS)
            return key.first > other.key.first;

        return key.second > other.key.second;
    }
};


// ========================================================
// D* LITE SAFE SEMANTIC PLANNER
// ========================================================

class DStarLitePlanner : public Planner {

private:

    // ----------------------------------------------------
    // Problem data
    // ----------------------------------------------------

    unordered_map<ID, State> stateMap;

    unordered_map<ID, Transition> transitionMap;

    // outgoing[state] = transition IDs
    unordered_map<ID, vector<ID>> outgoing;

    // incoming[state] = transition IDs
    unordered_map<ID, vector<ID>> incoming;

    unordered_set<ID> badStates;

    ID start = 0;
    ID goal = 0;

    // ----------------------------------------------------
    // D* Lite values
    // ----------------------------------------------------

    unordered_map<ID, double> g;
    unordered_map<ID, double> rhs;

    // Used by D* Lite when the start moves.
    double km = 0.0;

    ID lastStart = 0;

    priority_queue<QueueNode> openList;

    // ----------------------------------------------------
    // Statistics
    // ----------------------------------------------------

    size_t exploredStates = 0;


    // ====================================================
    // Utility: Euclidean distance
    // ====================================================

    double euclidean(
        ID a,
        ID b
    ) const {

        auto ita = stateMap.find(a);
        auto itb = stateMap.find(b);

        if (ita == stateMap.end() ||
            itb == stateMap.end()) {

            return INF;
        }

        const vector<double>& x =
            ita->second.embedding;

        const vector<double>& y =
            itb->second.embedding;

        if (x.size() != y.size())
            return INF;

        double sum = 0.0;

        for (size_t i = 0; i < x.size(); ++i) {
            double d = x[i] - y[i];
            sum += d * d;
        }

        return sqrt(sum);
    }


    // ====================================================
    // Heuristic
    // ====================================================

    double heuristic(
        ID a,
        ID b
    ) const {

        return euclidean(a, b);
    }


    // ====================================================
    // Check bad state
    // ====================================================

    bool isBad(ID s) const {
        return badStates.find(s) != badStates.end();
    }


    // ====================================================
    // Get g value
    // ====================================================

    double getG(ID s) const {

        auto it = g.find(s);

        if (it == g.end())
            return INF;

        return it->second;
    }


    // ====================================================
    // Get rhs value
    // ====================================================

    double getRHS(ID s) const {

        auto it = rhs.find(s);

        if (it == rhs.end())
            return INF;

        return it->second;
    }


    // ====================================================
    // Set g value
    // ====================================================

    void setG(
        ID s,
        double value
    ) {

        g[s] = value;
    }


    // ====================================================
    // Set rhs value
    // ====================================================

    void setRHS(
        ID s,
        double value
    ) {

        rhs[s] = value;
    }


    // ====================================================
    // Edge cost
    // ====================================================

    double edgeCost(
        ID transitionID
    ) const {

        auto it = transitionMap.find(transitionID);

        if (it == transitionMap.end())
            return INF;

        const Transition& t = it->second;

        if (!t.available)
            return INF;

        if (isBad(t.from) ||
            isBad(t.to)) {

            return INF;
        }

        if (t.cost < 0.0)
            return INF;

        return t.cost;
    }


    // ====================================================
    // Calculate RHS
    // ====================================================

    double calculateRHS(
        ID s
    ) const {

        if (s == goal)
            return 0.0;

        if (isBad(s))
            return INF;

        auto it = outgoing.find(s);

        if (it == outgoing.end())
            return INF;

        double best = INF;

        for (ID tid : it->second) {

            auto trIt = transitionMap.find(tid);

            if (trIt == transitionMap.end())
                continue;

            const Transition& tr =
                trIt->second;

            if (!tr.available)
                continue;

            if (isBad(tr.from) ||
                isBad(tr.to))
                continue;

            double c = edgeCost(tid);

            if (!isfinite(c))
                continue;

            double candidate =
                c + getG(tr.to);

            best = min(best, candidate);
        }

        return best;
    }


    // ====================================================
    // Calculate D* Lite key
    // ====================================================

    Key calculateKey(
        ID s
    ) const {

        double minValue =
            min(getG(s), getRHS(s));

        return {
            minValue + heuristic(start, s) + km,
            minValue
        };
    }


    // ====================================================
    // Push node to open list
    // ====================================================

    void pushOpen(
        ID s
    ) {

        openList.push({
            s,
            calculateKey(s)
        });
    }


    // ====================================================
    // Initialize D* Lite
    // ====================================================

    void initialize() {

        while (!openList.empty())
            openList.pop();

        g.clear();
        rhs.clear();

        km = 0.0;

        lastStart = start;

        setRHS(goal, 0.0);
        setG(goal, INF);

        pushOpen(goal);
    }


    // ====================================================
    // Update Vertex
    // ====================================================

    void updateVertex(
        ID u
    ) {

        if (u != goal) {

            if (isBad(u)) {

                setRHS(u, INF);

            } else {

                setRHS(
                    u,
                    calculateRHS(u)
                );
            }
        }

        if (fabs(getG(u) - getRHS(u)) > EPS) {

            pushOpen(u);
        }
    }


    // ====================================================
    // Compare keys
    // ====================================================

    bool keyLess(
        const Key& a,
        const Key& b
    ) const {

        if (a.first < b.first - EPS)
            return true;

        if (a.first > b.first + EPS)
            return false;

        return a.second < b.second - EPS;
    }


    // ====================================================
    // D* Lite main computation
    // ====================================================

    void computeShortestPath() {

        exploredStates = 0;

        while (!openList.empty()) {

            QueueNode top =
                openList.top();

            Key startKey =
                calculateKey(start);

            if (!keyLess(top.key, startKey) &&
                fabs(getRHS(start) -
                     getG(start)) <= EPS) {

                break;
            }

            openList.pop();

            ID u = top.state;

            Key oldKey = top.key;

            Key newKey =
                calculateKey(u);

            if (keyLess(oldKey, newKey)) {

                pushOpen(u);
                continue;
            }

            ++exploredStates;

            if (getG(u) >
                getRHS(u)) {

                setG(
                    u,
                    getRHS(u)
                );

                // Update predecessors.
                auto it =
                    incoming.find(u);

                if (it != incoming.end()) {

                    for (ID tid : it->second) {

                        auto trIt =
                            transitionMap.find(tid);

                        if (trIt == transitionMap.end())
                            continue;

                        updateVertex(
                            trIt->second.from
                        );
                    }
                }

            } else {

                setG(
                    u,
                    INF
                );

                updateVertex(u);

                auto it =
                    incoming.find(u);

                if (it != incoming.end()) {

                    for (ID tid : it->second) {

                        auto trIt =
                            transitionMap.find(tid);

                        if (trIt == transitionMap.end())
                            continue;

                        updateVertex(
                            trIt->second.from
                        );
                    }
                }
            }
        }
    }


    // ====================================================
    // Find nearest bad-state distance
    // ====================================================

    double nearestBadDistance(
        ID state
    ) const {

        if (badStates.empty())
            return INF;

        double best = INF;

        for (ID bad : badStates) {

            if (!stateMap.count(bad))
                continue;

            best = min(
                best,
                euclidean(state, bad)
            );
        }

        return best;
    }


    // ====================================================
    // Reconstruct optimal path
    // ====================================================

    PlanningResult reconstructPath() {

        PlanningResult result;

        if (!isfinite(getG(start))) {

            result.success = false;

            return result;
        }

        ID current = start;

        unordered_set<ID> visited;

        result.statePath.push_back(current);

        double totalCost = 0.0;

        double minimumSafety = INF;

        double reliability = 1.0;

        while (current != goal) {

            // Prevent cycles.
            if (visited.count(current)) {

                result.success = false;

                return result;
            }

            visited.insert(current);

            minimumSafety =
                min(
                    minimumSafety,
                    nearestBadDistance(current)
                );

            auto it =
                outgoing.find(current);

            if (it == outgoing.end()) {

                result.success = false;

                return result;
            }

            double bestValue = INF;

            ID bestTransition = 0;
            ID bestNext = 0;

            double bestSafety = -1.0;
            double bestReliability = -1.0;

            for (ID tid : it->second) {

                auto trIt =
                    transitionMap.find(tid);

                if (trIt == transitionMap.end())
                    continue;

                const Transition& tr =
                    trIt->second;

                if (!tr.available)
                    continue;

                if (isBad(tr.to))
                    continue;

                double c =
                    edgeCost(tid);

                if (!isfinite(c))
                    continue;

                double candidate =
                    c + getG(tr.to);

                double candidateSafety =
                    nearestBadDistance(tr.to);

                double candidateReliability =
                    tr.reliability;

                bool better = false;

                if (candidate <
                    bestValue - EPS) {

                    better = true;

                } else if (
                    fabs(candidate -
                         bestValue) <= EPS) {

                    if (candidateSafety >
                        bestSafety + EPS) {

                        better = true;

                    } else if (
                        fabs(candidateSafety -
                             bestSafety) <= EPS &&
                        candidateReliability >
                        bestReliability) {

                        better = true;
                    }
                }

                if (better) {

                    bestValue = candidate;

                    bestTransition = tid;

                    bestNext = tr.to;

                    bestSafety =
                        candidateSafety;

                    bestReliability =
                        candidateReliability;
                }
            }

            if (bestTransition == 0 &&
                current != goal) {

                result.success = false;

                return result;
            }

            const Transition& chosen =
                transitionMap.at(
                    bestTransition
                );

            totalCost += chosen.cost;

            reliability *=
                chosen.reliability;

            minimumSafety =
                min(
                    minimumSafety,
                    nearestBadDistance(bestNext)
                );

            result.transitionPath.push_back(
                bestTransition
            );

            current = bestNext;

            result.statePath.push_back(
                current
            );
        }

        result.success = true;

        result.totalCost = totalCost;

        result.safetyScore =
            minimumSafety;

        result.cumulativeReliability =
            reliability;

        result.exploredStates =
            exploredStates;

        return result;
    }


    // ====================================================
    // Build graph
    // ====================================================

    void buildGraph(
        const PlanningProblem& problem
    ) {

        stateMap.clear();

        transitionMap.clear();

        outgoing.clear();

        incoming.clear();

        badStates.clear();

        for (const State& s :
             problem.states) {

            stateMap[s.id] = s;
        }

        for (ID b :
             problem.badStates) {

            badStates.insert(b);
        }

        for (const Transition& t :
             problem.transitions) {

            transitionMap[t.id] = t;

            outgoing[t.from].push_back(
                t.id
            );

            incoming[t.to].push_back(
                t.id
            );
        }

        start =
            problem.initialState;

        goal =
            problem.goalState;
    }


public:

    // ====================================================
    // Constructor
    // ====================================================

    DStarLitePlanner() = default;


    // ====================================================
    // Standard plan interface
    // ====================================================

    PlanningResult plan(
        const PlanningProblem& problem
    ) override {

        auto startTime =
            chrono::high_resolution_clock::now();

        buildGraph(problem);

        PlanningResult result;

        if (!stateMap.count(start) ||
            !stateMap.count(goal)) {

            return result;
        }

        if (isBad(start) ||
            isBad(goal)) {

            return result;
        }

        initialize();

        computeShortestPath();

        result =
            reconstructPath();

        auto endTime =
            chrono::high_resolution_clock::now();

        result.planningTimeMs =
            chrono::duration<double, milli>(
                endTime - startTime
            ).count();

        result.memoryEstimateBytes =
            stateMap.size() * sizeof(State) +
            transitionMap.size() *
                sizeof(Transition) +
            g.size() * sizeof(double) +
            rhs.size() * sizeof(double);

        return result;
    }


    // ====================================================
    // Dynamic replanning
    // ====================================================

    PlanningResult replan(
        const PlanningProblem& problem
    ) {

        return plan(problem);
    }
};


// ========================================================
// PRINT RESULT
// ========================================================

void printResult(
    const string& title,
    const PlanningResult& result
) {

    cout << "\n";
    cout << "====================================================\n";
    cout << title << "\n";
    cout << "====================================================\n";

    cout << fixed
         << setprecision(4);

    cout << "Success                : "
         << (result.success ? "YES" : "NO")
         << "\n";

    if (!result.success) {

        cout << "No safe path exists.\n";

        cout << "Planning time (ms)     : "
             << result.planningTimeMs
             << "\n";

        return;
    }

    cout << "State Path             : ";

    for (size_t i = 0;
         i < result.statePath.size();
         ++i) {

        if (i > 0)
            cout << " -> ";

        cout << result.statePath[i];
    }

    cout << "\n";

    cout << "Transition Path        : ";

    for (size_t i = 0;
         i < result.transitionPath.size();
         ++i) {

        if (i > 0)
            cout << " -> ";

        cout << result.transitionPath[i];
    }

    cout << "\n";

    cout << "Total Cost             : "
         << result.totalCost
         << "\n";

    cout << "Minimum Safety Distance: "
         << result.safetyScore
         << "\n";

    cout << "Cumulative Reliability : "
         << result.cumulativeReliability
         << "\n";

    cout << "Explored States        : "
         << result.exploredStates
         << "\n";

    cout << "Planning Time (ms)     : "
         << result.planningTimeMs
         << "\n";

    cout << "Memory Estimate (bytes): "
         << result.memoryEstimateBytes
         << "\n";
}


// ========================================================
// CREATE TEST CASE 1
// ========================================================

PlanningProblem testCase1() {

    PlanningProblem p;

    p.initialState = 0;
    p.goalState = 3;

    p.states = {
        State(0, {0, 0}),
        State(1, {1, 0}),
        State(2, {2, 0}),
        State(3, {3, 0})
    };

    p.transitions = {

        Transition(
            1, 0, 1,
            1.0, 1.0, 0.99, true
        ),

        Transition(
            2, 1, 2,
            1.0, 1.0, 0.99, true
        ),

        Transition(
            3, 2, 3,
            1.0, 1.0, 0.99, true
        )
    };

    return p;
}


// ========================================================
// CREATE TEST CASE 2
// ========================================================

PlanningProblem testCase2() {

    PlanningProblem p;

    p.initialState = 0;
    p.goalState = 5;

    p.states = {

        State(0, {0, 0}),   // S
        State(1, {1, 0}),   // A
        State(2, {2, 0}),   // X bad
        State(3, {0, 1}),   // C
        State(4, {1, 1}),   // D
        State(5, {2, 1})    // G
    };

    p.badStates = {
        2
    };

    p.transitions = {

        // S -> A -> X -> G
        Transition(
            1, 0, 1,
            1.0, 1.0, 0.95, true
        ),

        Transition(
            2, 1, 2,
            1.0, 1.0, 0.95, true
        ),

        Transition(
            3, 2, 5,
            1.0, 1.0, 0.95, true
        ),

        // S -> C -> D -> G
        Transition(
            4, 0, 3,
            1.0, 1.0, 0.99, true
        ),

        Transition(
            5, 3, 4,
            1.0, 1.0, 0.99, true
        ),

        Transition(
            6, 4, 5,
            1.0, 1.0, 0.99, true
        )
    };

    return p;
}


// ========================================================
// CREATE TEST CASE 3
// ========================================================

PlanningProblem testCase3() {

    PlanningProblem p;

    p.initialState = 0;
    p.goalState = 7;

    p.states = {

        State(0, {0, 0}),     // S

        // Short but unsafe route
        State(1, {1, 0}),
        State(2, {2, 0}),
        State(3, {3, 0}),

        // Longer and safer route
        State(4, {0, 3}),
        State(5, {1, 3}),
        State(6, {2, 3}),

        State(7, {3, 3})      // G
    };

    // Bad state near short route.
    p.badStates = {
        2
    };

    p.transitions = {

        // Short path
        Transition(
            1, 0, 1,
            1.0, 0.5, 0.95, true
        ),

        Transition(
            2, 1, 2,
            1.0, 0.4, 0.95, true
        ),

        Transition(
            3, 2, 3,
            1.0, 0.4, 0.95, true
        ),

        Transition(
            4, 3, 7,
            1.0, 0.4, 0.95, true
        ),

        // Safer path
        Transition(
            5, 0, 4,
            2.0, 1.0, 0.99, true
        ),

        Transition(
            6, 4, 5,
            2.0, 1.0, 0.99, true
        ),

        Transition(
            7, 5, 6,
            2.0, 1.0, 0.99, true
        ),

        Transition(
            8, 6, 7,
            2.0, 1.0, 0.99, true
        )
    };

    return p;
}


// ========================================================
// CREATE TEST CASE 4
// ========================================================

PlanningProblem testCase4() {

    PlanningProblem p;

    p.initialState = 0;
    p.goalState = 3;

    p.states = {

        State(0, {0, 0}), // S
        State(1, {1, 0}), // A
        State(2, {0, 1}), // B
        State(3, {2, 0})  // G
    };

    p.transitions = {

        // Initial shortest path
        Transition(
            1, 0, 1,
            1.0, 1.0, 0.99, true
        ),

        Transition(
            2, 1, 3,
            1.0, 1.0, 0.99, true
        ),

        // Alternative
        Transition(
            3, 0, 2,
            2.0, 1.0, 0.99, true
        ),

        Transition(
            4, 2, 3,
            2.0, 1.0, 0.99, true
        )
    };

    return p;
}


// ========================================================
// CREATE TEST CASE 5
// ========================================================

PlanningProblem testCase5() {

    PlanningProblem p;

    p.initialState = 0;
    p.goalState = 4;

    p.states = {

        State(0, {0, 0}),
        State(1, {1, 0}),
        State(2, {2, 0}),
        State(3, {1, 1}),
        State(4, {2, 1})
    };

    p.transitions = {

        Transition(
            1, 0, 1,
            1.0, 1.0, 0.99, true
        ),

        Transition(
            2, 1, 2,
            1.0, 1.0, 0.99, true
        ),

        Transition(
            3, 0, 3,
            1.0, 1.0, 0.99, true
        ),

        Transition(
            4, 3, 4,
            1.0, 1.0, 0.99, true
        ),

        Transition(
            5, 2, 4,
            1.0, 1.0, 0.99, true
        )
    };

    return p;
}


// ========================================================
// CREATE TEST CASE 6
// ========================================================

PlanningProblem testCase6() {

    PlanningProblem p;

    p.initialState = 0;
    p.goalState = 5;

    p.states = {

        State(0, {0, 0}),
        State(1, {1, 0}),
        State(2, {2, 0}),
        State(3, {3, 0}),
        State(4, {4, 0}),
        State(5, {5, 0})
    };

    p.transitions = {

        Transition(
            1, 0, 1,
            2.0, 1.0, 0.95, true
        ),

        Transition(
            2, 1, 2,
            2.0, 1.0, 0.95, true
        ),

        Transition(
            3, 2, 3,
            2.0, 1.0, 0.95, true
        ),

        Transition(
            4, 3, 4,
            2.0, 1.0, 0.95, true
        ),

        Transition(
            5, 4, 5,
            2.0, 1.0, 0.95, true
        )
    };

    return p;
}


// ========================================================
// MAIN
// ========================================================

int main() {

    cout << "\n";
    cout << " Safe Semantic Planner using D* Lite\n";
    cout << "====================================================\n";
    


    DStarLitePlanner planner;


    // ----------------------------------------------------
    // Test Case 1
    // ----------------------------------------------------

    {
        PlanningProblem p =
            testCase1();

        PlanningResult result =
            planner.plan(p);

        printResult(
            "TEST CASE 1 - BASIC REACHABILITY",
            result
        );
    }


    // ----------------------------------------------------
    // Test Case 2
    // ----------------------------------------------------

    {
        PlanningProblem p =
            testCase2();

        PlanningResult result =
            planner.plan(p);

        printResult(
            "TEST CASE 2 - BAD STATE AVOIDANCE",
            result
        );
    }


    // ----------------------------------------------------
    // Test Case 3
    // ----------------------------------------------------

    {
        PlanningProblem p =
            testCase3();

        PlanningResult result =
            planner.plan(p);

        printResult(
            "TEST CASE 3 - SAFETY MARGIN",
            result
        );
    }


    // ----------------------------------------------------
    // Test Case 4
    // ----------------------------------------------------

    {
        PlanningProblem p =
            testCase4();

        PlanningResult result1 =
            planner.plan(p);

        printResult(
            "TEST CASE 4A - BEFORE TRANSITION FAILURE",
            result1
        );


        // Make A -> G unavailable.
        for (auto& t :
             p.transitions) {

            if (t.from == 1 &&
                t.to == 3) {

                t.available = false;
            }
        }


        PlanningResult result2 =
            planner.replan(p);

        printResult(
            "TEST CASE 4B - AFTER TRANSITION FAILURE",
            result2
        );
    }


    // ----------------------------------------------------
    // Test Case 5
    // ----------------------------------------------------

    {
        PlanningProblem p =
            testCase5();

        PlanningResult result1 =
            planner.plan(p);

        printResult(
            "TEST CASE 5A - ORIGINAL GOAL",
            result1
        );


        // Change goal.
        p.goalState = 2;

        PlanningResult result2 =
            planner.replan(p);

        printResult(
            "TEST CASE 5B - AFTER GOAL UPDATE",
            result2
        );
    }


    // ----------------------------------------------------
    // Test Case 6
    // ----------------------------------------------------

    {
        PlanningProblem p =
            testCase6();

        PlanningResult result1 =
            planner.plan(p);

        printResult(
            "TEST CASE 6A - BEFORE SHORTCUT",
            result1
        );


        // Add a new shortcut.
        p.transitions.push_back(
            Transition(
                100,
                0,
                5,
                2.0,
                1.0,
                0.99,
                true
            )
        );


        PlanningResult result2 =
            planner.replan(p);

        printResult(
            "TEST CASE 6B - AFTER TRANSITION ADDITION",
            result2
        );
    }


    cout << "\n";
    cout << "====================================================\n";
    cout << " All test cases completed.\n";
    cout << "====================================================\n";

    return 0;
}