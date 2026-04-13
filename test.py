# after training, print most visited states
import pickle
import numpy as np

with open("weights_ql.pkl", "rb") as f:
    Q = pickle.load(f)

print(f"Total states: {len(Q)}")

# print Q values for all-zero state
zero_state = tuple([0]*18)
if zero_state in Q:
    print("All-zero state Q:", Q[zero_state])
else:
    print("All-zero state never visited")