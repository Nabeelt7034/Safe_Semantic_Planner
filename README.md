# PCCST503 – Machine Learning
## Assignment 1 – Design of a Safe Semantic Planner in a Finite Cartesian State Space

### Student Details

**Name:** Nabeel T  
**Register Number:** LTCR24CS075  
**Course:** PCCST503 – Machine Learning  
**Department:** Computer Science and Engineering  

---

## 1. Assignment Overview

This project implements a Safe Semantic Planner for a finite Cartesian state space.

The planner computes a path from an initial state to a goal state while avoiding predefined bad states. Each state is represented using a Cartesian embedding, and each directed transition contains information about cost, safety, reliability, and availability.

The project focuses on:

- Graph search
- Heuristic design
- Safe path planning
- Optimization
- Dynamic replanning
- Experimental evaluation

---

## 2. Problem Definition

Let:

S = {s1, s2, ..., sn}

be a finite set of states embedded in a Cartesian space R^d.

Each state has a vector representation:

s_i = (x_1, x_2, ..., x_d)

The planner receives:

- Initial state
- Goal state
- Set of bad states
- Directed transitions

Each transition contains:

- Transition cost
- Reliability
- Safety score
- Availability flag

The planner must find a valid path that:

1. Reaches the goal state.
2. Never visits a bad state.
3. Minimizes total transition cost.
4. Maximizes the minimum Euclidean distance from visited states to the nearest bad state.
5. Produces the solution within reasonable execution time.

---

## 3. Selected Algorithm

### D* Lite

This project uses the **D* Lite** algorithm.

D* Lite is an incremental heuristic search algorithm suitable for environments where the graph or goal can change during execution.

It maintains:

- `g(s)` values
- `rhs(s)` values
- A priority queue
- Heuristic information

The algorithm can be used to replan when:

- A transition becomes unavailable.
- A new transition is added.
- The goal changes.
- The set of bad states changes.

---

## 4. State Representation

Each state contains:

```text
State
├── id
└── embedding