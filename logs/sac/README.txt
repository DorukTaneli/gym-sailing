---jtr-v0: Uses Roman's reward
jtr-v0_1: trained for 40000 steps with default sb3 hyperparameters (nr)

---jtr-v1: New reward capped at 1000
jtr-v1_2: trained for 100000 steps with default sb3 hyperparameters (nr)
jtr-v1_5: (test)trained for 100 steps with gsde hyperparameters
jtr-v1_6: trained for 1e5 steps with gsde hyperparams, took 41 hours

(nr): not runnable due to versioning inconsistencies

---jtr-v2: v1 parallelized - no runs as it is slower than v1

---jtr-modelless-v0: no models, fixed speed, calculate heading from rudder mathematically
	-300 reward penalty for not reaching the goal, reset if taking too long to reach goal
jtr-modelless-v0_4: default hyperparameters, 1e5 steps, last reward: 587.58 +/- 157.73
jtr-modelless-v0_5: gSDE hyperparameters, 1e5 steps

---jtr-modelless-v1: simplify the observation space to 4 features:
			speed, angle, relative lat, relative long
jtr-modelless-v1_1: default hyperparameters, 1e5 steps
jtr-modelless-v1_2: gSDE hyperparameters, 1e5 steps

---jtr-modelless-v2: change reward function: substract reward when moving away from goal
			add penalty for rudder angles (discourage high rudder movements)
jtr-modelless-v2_1: default parameters trained for 1m steps, 
		    still can't reach goal consistently

---jtr-modelless-v3: further simplify observation space.
			only 2 features: current heading and heading to goal
jtr-modelless-v3_2: default hp, trained for 1e4 steps. Can reach goal consistently!	

---jtr-modelless-v4: update reward, no rudder penalties as they are necessary irl
			reaching goal: 200, not finishing: -50, finishing like original: -100
			solved is >100, can reach goal faster than original boat
jtr-modelless-v4_1: default hp, 1e5 steps. can reach goal consistently, converges in 12k steps
