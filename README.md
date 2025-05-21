# 🌱 Greenhouse Plant Growth with Reinforcement Learning

<p align="center">
  <img src="Sims-GYM/Graph_elements/Growth/Growth_5.png" alt="Greenhouse GUI Demo" width="300"/>
</p>

<p align="center"><i>Interactive GUI to visualize plant growth during RL training</i></p>
This project implements a **Reinforcement Learning (RL)** framework for managing temperature and humidity in a greenhouse, guiding a single plant through different growth stages up to harvest.
> **Visual RL Environment** – This project features an interactive **GUI** for visualizing plant growth in real time as the RL agent acts in the environment.
---

## 🎯 Objective

The agent controls temperature and humidity to:
- Maximize plant growth across **5 stages** (germination → harvest)
- Learn an optimal policy using **Monte Carlo (MC) methods**
- Optionally, minimize energy cost by penalizing excessive action use

---

## 🌡️ Environment Description

- **State**: A triplet `(growth_stage, temperature, humidity)`
  - `growth_stage` ∈ {0,1,2,3,4,5}
  - `temperature`, `humidity` ∈ [0, 10]
- **Actions**:
  - `0`: No action  
  - `1`: Increase temperature by 1  
  - `2`: Increase humidity by 1

- **Stochastic decay**:
  - With probability `1 - pt`, temperature may drop by 1
  - With probability `1 - ph`, humidity may drop by 1

- **Growth dynamics**:
  - **+1** if both temp and hum are optimal (green)
  - **0** if at least one is good (yellow)
  - **−1** if at least one is poor (orange)
  - **−2** if at least one is very poor (red)

- **Reward**:
  - `+100` if the plant reaches `growth = 5` (harvest)
  - Optionally: `−6` for increasing temperature, `−5` for increasing humidity

---

## 🧠 RL Approach

- **Epsilon-Greedy** exploration
- **Monte Carlo estimation** of:
  - State-Value function `V(s)`
  - Action-Value function `Q(s, a)`
- **Tabular policy** updated using greedy improvement

---

## 📂 Project Structure
📦 project-root/
- RL_utils.py # RL core logic, episode handling, MC learning
- RL_plot_utils.py # Dynamic and static visualization tools (Plotly & Matplotlib)
- RL-Main.ipynb # Main notebook to train, simulate and visualize the policy
- Plant_Growth_Project.ipynb # Additional analysis and experiments
- Greenhouse Management.pdf # Problem description and theoretical formulation


---

## 📊 Visualizations

- `plot_V_single()`, `plot_value_dyn()`  
  → visualize the value function per growth stage as a heatmap  
- `plot_policy_dyn()`  
  → slider-based view of the learned policy  
- `training_plot()`, `training_plot2()`  
  → smoothed reward and episode length over training  
- `plot_Qs()`  
  → shows Q-values per action across conditions  
- `plot_data_conditions()`  
  → compares value, policy, and actual growth potential

---

## ▶️ Getting Started

### 1. Install dependencies
```bash
pip install numpy matplotlib plotly pygame

