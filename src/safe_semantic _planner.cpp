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

using namespace std;

using ID = uint64_t;
const double INF = numeric_limits<double>::infinity();

struct State {
    ID id{};
    vector<double> embedding;
};

struct Transition {
    ID id{};
    ID from{};
    ID to{};
    double cost{1.0};
    double safety{1.0};
    double reliability{1.0};
    bool available{true};
};

struct PlanningProblem {
    ID initialState{};
    ID goalState{};
    vector<ID> badStates;
    vector<State> states;
    vector<Transition> transitions;
};

struct PlanningResult {
    bool success{false};
    vector<ID> statePath;
    vector<ID> transitionPath;
    double totalCost{INF};
    double safetyScore{0.0}; // minimum Euclidean distance to a bad state
    double cumulativeReliability{0.0};
    size_t exploredStates{0};
    double planningTimeMs{0.0};
};

class Planner {
public:
    virtual PlanningResult plan(const PlanningProblem& problem) = 0;
    virtual ~Planner() = default;
};

/*
 * SafeSemanticPlanner
 *
 * Uses an A*-style graph search with:
 *   f(n) = g(n) + h(n)
 *   h(n) = Euclidean distance from n to the goal.
 *
 * Safety is a hard constraint: bad states are never inserted into the search.
 * Among feasible paths, the planner minimizes a lexicographic objective:
 *   1) total transition cost
 *   2) maximize minimum Euclidean distance to a bad state
 *   3) maximize cumulative reliability
 *
 * The implementation is deliberately generic and supports dynamic updates by
 * replacing the PlanningProblem and calling plan() again. The search structures
 * are rebuilt per call; this is a simple baseline suitable for the assignment.
 * A production LPA*/D* Lite version can reuse g/rhs values between updates.
 */
class SafeSemanticPlanner : public Planner {
    unordered_map<ID, State> stateMap;
    unordered_map<ID, vector<const Transition*>> outgoing;
    unordered_set<ID> bad;
    ID goal{};
    vector<double> goalEmbedding;

    double euclidean(const vector<double>& a, const vector<double>& b) const {
        if (a.size() != b.size()) return INF;
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return sqrt(sum);
    }

    double distanceToNearestBad(ID s) const {
        if (bad.empty()) return INF;
        const auto& e = stateMap.at(s).embedding;
        double best = INF;
        for (ID b : bad) {
            if (stateMap.count(b))
                best = min(best, euclidean(e, stateMap.at(b).embedding));
        }
        return best;
    }

    double heuristic(ID s) const {
        return euclidean(stateMap.at(s).embedding, goalEmbedding);
    }

    struct Node {
        ID id;
        double f;
        double g;
        bool operator>(const Node& other) const { return f > other.f; }
    };

public:
    PlanningResult plan(const PlanningProblem& p) override {
        auto t0 = chrono::high_resolution_clock::now();
        PlanningResult result;

        stateMap.clear();
        outgoing.clear();
        bad.clear();

        for (const auto& s : p.states) stateMap[s.id] = s;
        for (ID b : p.badStates) bad.insert(b);
        goal = p.goalState;

        if (!stateMap.count(p.initialState) || !stateMap.count(goal) ||
            bad.count(p.initialState) || bad.count(goal)) {
            result.planningTimeMs =
                chrono::duration<double, milli>(chrono::high_resolution_clock::now() - t0).count();
            return result;
        }

        goalEmbedding = stateMap.at(goal).embedding;

        for (const auto& tr : p.transitions) {
            if (tr.available && stateMap.count(tr.from) && stateMap.count(tr.to))
                outgoing[tr.from].push_back(&tr);
        }

        unordered_map<ID, double> gScore, minSafety;
        unordered_map<ID, double> reliability;
        unordered_map<ID, ID> parentState, parentTransition;

        priority_queue<Node, vector<Node>, greater<Node>> open;
        unordered_set<ID> closed;

        gScore[p.initialState] = 0.0;
        minSafety[p.initialState] = distanceToNearestBad(p.initialState);
        reliability[p.initialState] = 1.0;

        open.push({p.initialState, heuristic(p.initialState), 0.0});

        while (!open.empty()) {
            Node cur = open.top();
            open.pop();

            if (closed.count(cur.id)) continue;
            closed.insert(cur.id);
            ++result.exploredStates;

            if (cur.id == goal) {
                result.success = true;
                ID x = goal;
                result.totalCost = gScore[x];
                result.safetyScore = minSafety[x];
                result.cumulativeReliability = reliability[x];

                vector<ID> revStates{goal};
                vector<ID> revTransitions;
                while (x != p.initialState) {
                    revTransitions.push_back(parentTransition.at(x));
                    x = parentState.at(x);
                    revStates.push_back(x);
                }
                reverse(revStates.begin(), revStates.end());
                reverse(revTransitions.begin(), revTransitions.end());
                result.statePath = move(revStates);
                result.transitionPath = move(revTransitions);
                break;
            }

            for (const Transition* tr : outgoing[cur.id]) {
                ID v = tr->to;
                if (bad.count(v) || closed.count(v)) continue;

                double tentativeG = gScore[cur.id] + tr->cost;

                bool betterCost = !gScore.count(v) || tentativeG < gScore[v] - 1e-12;

                // Tie-breaking favors a safer route and then higher reliability.
                double candidateSafety = min(minSafety[cur.id], distanceToNearestBad(v));
                double candidateReliability = reliability[cur.id] * tr->reliability;

                bool tieBetter =
                    gScore.count(v) &&
                    fabs(tentativeG - gScore[v]) <= 1e-12 &&
                    (candidateSafety > minSafety[v] + 1e-12 ||
                     (fabs(candidateSafety - minSafety[v]) <= 1e-12 &&
                      candidateReliability > reliability[v] + 1e-12));

                if (betterCost || tieBetter) {
                    gScore[v] = tentativeG;
                    minSafety[v] = candidateSafety;
                    reliability[v] = candidateReliability;
                    parentState[v] = cur.id;
                    parentTransition[v] = tr->id;
                    open.push({v, tentativeG + heuristic(v), tentativeG});
                }
            }
        }

        result.planningTimeMs =
            chrono::duration<double, milli>(chrono::high_resolution_clock::now() - t0).count();
        return result;
    }
};

static void printResult(const PlanningResult& r) {
    cout << fixed << setprecision(3);
    cout << "Success: " << (r.success ? "YES" : "NO") << '\n';
    cout << "State path: ";
    if (r.success) {
        for (size_t i = 0; i < r.statePath.size(); ++i) {
            if (i) cout << " -> ";
            cout << r.statePath[i];
        }
    }
    cout << '\n';
    cout << "Transition path: ";
    if (r.success) {
        for (size_t i = 0; i < r.transitionPath.size(); ++i) {
            if (i) cout << " -> ";
            cout << r.transitionPath[i];
        }
    }
    cout << '\n';
    cout << "Total cost: " << r.totalCost << '\n';
    cout << "Minimum safety distance: " << r.safetyScore << '\n';
    cout << "Cumulative reliability: " << r.cumulativeReliability << '\n';
    cout << "Explored states: " << r.exploredStates << '\n';
    cout << "Planning time (ms): " << r.planningTimeMs << '\n';
}

static PlanningProblem buildDemoProblem() {
    PlanningProblem p;
    p.initialState = 0;
    p.goalState = 5;

    p.states = {
        {0, {0, 0}}, {1, {1, 0}}, {2, {2, 0}},
        {3, {0, 1}}, {4, {1, 1}}, {5, {2, 1}},
        {6, {2, 2}}
    };

    p.badStates = {2};

    p.transitions = {
        {0, 0, 1, 1, 1.0, 0.99, true},
        {1, 1, 2, 1, 1.0, 0.99, true}, // enters bad state: ignored
        {2, 0, 3, 1, 1.0, 1.00, true},
        {3, 3, 4, 1, 1.0, 0.99, true},
        {4, 4, 5, 1, 1.0, 0.99, true},
        {5, 1, 4, 2, 1.0, 0.98, true},
        {6, 4, 6, 1, 1.0, 0.98, true},
        {7, 6, 5, 1, 1.0, 0.98, true}
    };
    return p;
}

int main() {
    SafeSemanticPlanner planner;
    PlanningProblem problem = buildDemoProblem();

    cout << "PCCST503 - Safe Semantic Planner\n";
    cout << "Student: Nabeel T | Reg No: LTCR24CS075\n\n";
    cout << "Initial planning:\n";
    auto r1 = planner.plan(problem);
    printResult(r1);

    // Dynamic update example: transition 4->5 becomes unavailable.
    for (auto& tr : problem.transitions)
        if (tr.from == 4 && tr.to == 5) tr.available = false;

    cout << "\nAfter dynamic transition update (4 -> 5 unavailable):\n";
    auto r2 = planner.plan(problem);
    printResult(r2);

    // Goal update example.
    problem.goalState = 6;
    cout << "\nAfter goal update (goal = 6):\n";
    auto r3 = planner.plan(problem);
    printResult(r3);

    return 0;
}
