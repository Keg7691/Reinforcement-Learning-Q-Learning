#!/usr/bin/env python
# coding: utf-8

# In[7]:


from collections import defaultdict
from matplotlib import pyplot as plt
import gymnasium as gym
import numpy as np
import pandas as pd
import time

class CliffWalkingAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        visualize_training: bool = False,
        render_frequency: int = 1 # nur bei visualize_training=True relevant
    ):
        """
        Q-Learning Agent für CliffWalking-v1.
        Reward: Each time step incurs -1 reward, unless the player stepped into the cliff, which incurs -100 reward.
        """
        # Initialisierungs Parameter - Umgebung

        self.render_env = env

        # Wenn Visualisierung nicht gewünscht ist, erstelle eine Trainingsumgebung ohne render_mode
        # ansonsten verwende die übergebene Umgebung für Training und Rendern
        if not visualize_training:
            try:
                self.train_env = gym.make(self.render_env.spec.id)
            except Exception:
                self.train_env = self.render_env
        else:
            # Visualisierung gewünscht: trainiere direkt in der gegebenen Umgebung
            self.train_env = self.render_env

        self.env = self.train_env

        # Intialisierung - Q-Table und Parameter

        # Q-Table
        self.q_values = defaultdict(lambda: np.zeros(self.train_env.action_space.n)) # hier noch mal anschauen

        # Parameter
        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Epsilon-Greedy
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Tracking
        self.training_error = []
        self.return_queue = []
        self.length_queue = []

        # Visualisierung
        self.visualize_training = visualize_training
        self.render_frequency = render_frequency

    # ---------------------------------------------------------

    def get_action(self, state: int) -> int:
        """Epsilon-Greedy Aktionswahl."""
        if np.random.random() < self.epsilon:
            # Verwende die Trainingsumgebung zum Sample, damit kein Render-Fenster erzeugt wird
            return self.train_env.action_space.sample()
        return int(np.argmax(self.q_values[state]))

    # ---------------------------------------------------------

    def update(self, state, action, reward, terminated, next_state):
        """
        Q-Learning Update mit Bellman-Gleichung.
        """
        future_q = 0 if terminated else np.max(self.q_values[next_state])

        target = reward + self.discount_factor * future_q
        temporal_diff = target - self.q_values[state][action]

        self.q_values[state][action] += self.lr * temporal_diff

        self.training_error.append(temporal_diff)

    # ---------------------------------------------------------

    def decay_epsilon(self):
        """Reduzierung von Epsilon pro Episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    # ---------------------------------------------------------

    def train_q_learning(self, episodes: int, max_steps: int = 200):
        """
        Training mit optionaler Visualisierung über rgb_array.
        Gibt bei visualize_training=True die gesammelten Frames zurück.
        """
        frames = []  # Sammlung für Jupyter-Animation

        for ep in range(episodes):
            state, _ = self.train_env.reset()

            total_reward = 0
            steps = 0

            for step in range(max_steps):

                # Frame speichern (nur wenn visualisierung eingeschaltet & Frequenz passt)
                if self.visualize_training and ep % self.render_frequency == 0:
                    try:
                        frame = self.render_env.render()
                        frames.append(frame)
                    except Exception:
                        pass

                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.train_env.step(action)

                done = terminated or truncated

                # Q-Learning Update
                self.update(state, action, reward, done, next_state)

                state = next_state
                total_reward += reward
                steps += 1

                if done:
                    break

            # Tracking
            self.return_queue.append(total_reward)
            self.length_queue.append(steps)
            self.decay_epsilon()

            if not self.visualize_training and ep % 100 == 0:
                print(f"[Episode {ep}] Reward={total_reward}, Steps={steps}, ε={self.epsilon:.3f}")

        print("Training abgeschlossen!\n")

        # Frames zurückgeben
        if self.visualize_training:
            return frames
        return None


    # ---------------------------------------------------------

    def run_policy_demo(self, max_steps=200):
        """
        Führt eine Demo-Episode aus und liefert Frames im rgb_array-Format.
        Ideal für Jupyter Notebook Animation.
        """
        print("\nStarte Demo-Episode basierend auf gelernten Q-Werten ...\n")

        # Demo-Umgebung immer mit rgb_array erzeugen
        demo_env = gym.make(self.render_env.spec.id, render_mode="rgb_array")

        state, _ = demo_env.reset()

        frames = []
        try:
            frames.append(demo_env.render())
        except:
            pass

        total_reward = 0

        for _ in range(max_steps):
            action = np.argmax(self.q_values[state])
            next_state, reward, terminated, truncated, _ = demo_env.step(action)

            # Frame speichern
            try:
                frame = demo_env.render()
                frames.append(frame)
            except:
                pass

            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        print(f"Demo abgeschlossen! Total Reward: {total_reward}")

        return frames


    # ---------------------------------------------------------
    def get_q_table_df(self):
        """Gibt die Q-Table als pandas DataFrame zurück."""
        df_q = pd.DataFrame(
            {state: self.q_values[state] for state in self.q_values}
        ).T
        df_q.columns = [f"action_{i}" for i in range(df_q.shape[1])]
        df_q.index.name = "state"

        return df_q

