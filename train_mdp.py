from mdp import MDP

# === Train the MDP components via Policy Iteration ===
verbose = True
path = "data-mini"
save_path = "saved-models"
k = 2

# Generate models whose n-gram values change from 1...k
for i in range(1, k+1):
    mm = MDP(path=path, k=i, verbose=verbose, save_path=save_path)
    mm.initialise_mdp()
    # Run the policy iteration and save the model
    mm.policy_iteration(max_iteration=1000)

# === Test ===
rs = MDP(path='data-mini', k=2)
rs.load('mdp-model_k=' + str(2) + '.pkl')