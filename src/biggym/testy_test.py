import numpy as np
from rewards.matsim import SimpleMATSimTraceScorer
import sys
import matplotlib.pyplot as plt

# matrix is 3 x 15
q = np.array([[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
              [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43],
              [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44]])

# print(q)
# print(q.shape)
#
# new_q = (q.reshape(9, 5, order="F"))  # q[:, ::3]
# print(new_q)


# [[ 0  9  18  27 36]
#  [ 1  10  19 28 37]
#  [ 2  11 20 29 38]
#  [ 3 12 21 30 39]
#  [ 4 13 22 31 40]
#  [ 5 14 23 32 41]
#  [ 6 15 24 33 42]
#  [ 7 16 25 34 43]
#  [ 8 17 26 35 44]]

trace = [[0, 6.5, 0],
         [2, 0.5, 10.0],
         [1, 9.25, 0],
         [2, 0.75, 10.0],
         [0, 7.0, 0]]

trace_2 = [[0, 24.0, 0]]

observation_space_mapping = {
    0: "act:home",
    1: "act:work",
    2: "trip:car",
}

trace_scorer = SimpleMATSimTraceScorer()
print(f"MATSIM Trace 1 Total : {trace_scorer.score(trace=trace, obs_map=observation_space_mapping)}")
print(f"MATSIM Trace 2 Total : {trace_scorer.score(trace=trace_2, obs_map=observation_space_mapping)}")


def normalise(val):
    min = -120  # TODO just random heuristic count for now
    max = 230
    return (val - min) / (max - min)

trace_1_tot = 0
trace_2_tot = 0
trace_1_list = []
trace_2_list = []
trace_1_sep_list = []
trace_2_sep_list = []
last_1_score = 0
last_2_score = 0
for step in range(96):
    submatrix = []
    normal_step = (step + 1) * 0.25

    # trace_2_sep_list.append(trace_scorer.score(trace=[[0, normal_step, 0]], obs_map=observation_space_mapping))  #  - last_2_score)
    score = trace_scorer.score(trace=[[0, normal_step, 0]], obs_map=observation_space_mapping)
    score = normalise(score)
    score = np.exp(score)
    trace_2_sep_list.append(score)
    last_2_score = trace_scorer.score(trace=[[0, normal_step, 0]], obs_map=observation_space_mapping)
    trace_2_tot += score

    tot_duration = 0
    for row in trace:
        value, duration, another_value = row
        if normal_step <= duration:
            submatrix.append([value, normal_step, another_value])
            break
        else:
            normal_step -= duration
            submatrix.append(row)

    # trace_1_sep_list.append(trace_scorer.score(trace=submatrix, obs_map=observation_space_mapping))  #  - last_1_score)
    score = trace_scorer.score(trace=submatrix, obs_map=observation_space_mapping)
    score = normalise(score)
    score = np.exp(score)
    trace_1_sep_list.append(score)
    last_1_score = trace_scorer.score(trace=submatrix, obs_map=observation_space_mapping)
    trace_1_tot += score

    # print(f"Trace 1 Total : {trace_1_tot}")
    # print(f"Trace 2 Total : {trace_2_tot}")
    # print("NEW LINE")
    trace_1_list.append(trace_1_tot)
    trace_2_list.append(trace_2_tot)

print(f"Trace 1 Total : {trace_1_tot}, the RL found one is 6162.573")
print(f"Trace 2 Total : {trace_2_tot}, the RL found one is 9318.882")

# plt.plot(trace_1_list)
# plt.plot(trace_2_list)
# plt.xticks(ticks=np.linspace(0, 96, 9), labels=[0, 3, 6, 9, 12, 15, 18, 21, 24])
# plt.show()
plt.plot(trace_1_sep_list)
plt.plot(trace_2_sep_list)
plt.xticks(ticks=np.linspace(0, 96, 9), labels=[0, 3, 6, 9, 12, 15, 18, 21, 24])
plt.show()

print(np.min([np.min(trace_1_sep_list), np.min(trace_2_sep_list)]))
print(np.max([np.max(trace_1_sep_list), np.max(trace_2_sep_list)]))
