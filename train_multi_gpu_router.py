#!/usr/bin/env python3
"""
Training Script for Multi-GPU RL Router

This script trains a Proximal Policy Optimization (PPO) agent to learn
optimal request routing across multiple GPUs for LLM inference.

Training Time: 4-8 hours on CPU, 1-2 hours on GPU
Expected Improvement: +5% scaling efficiency (87.5% → 92.5% on 4 GPUs)

Usage:
    python3 train_multi_gpu_router.py --num-gpus 4 --timesteps 200000

    Optional arguments:
    --num-gpus: Number of GPUs to simulate (default: 4)
    --timesteps: Training timesteps (default: 200000)
    --save-path: Where to save trained model (default: multi_gpu_router.zip)
    --log-dir: TensorBoard log directory (default: ./logs/multi_gpu_router)
"""

import argparse
import sys
import os


def train_multi_gpu_router(
    num_gpus: int = 4,
    total_timesteps: int = 200_000,
    save_path: str = "multi_gpu_router.zip",
    log_dir: str = "./logs/multi_gpu_router",
    n_envs: int = 4,
    eval_freq: int = 10_000,
    device: str = "cpu"
):
    """
    Train PPO agent for multi-GPU request routing.

    Args:
        num_gpus: Number of GPUs to simulate
        total_timesteps: Total training steps
        save_path: Path to save trained model
        log_dir: TensorBoard log directory
        n_envs: Number of parallel environments
        eval_freq: Evaluation frequency
        device: 'cpu' or 'cuda'
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("ERROR: stable-baselines3 not installed.")
        print("Install with: pip install stable-baselines3")
        sys.exit(1)

    from memopt.rl_router_env import MultiGPURouterEnv

    print("="*70)
    print("MULTI-GPU RL ROUTER TRAINING")
    print("="*70)
    print(f"Number of GPUs: {num_gpus}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Device: {device}")
    print(f"Log directory: {log_dir}")
    print(f"Save path: {save_path}")
    print("="*70)

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Create environment factory
    def make_env(rank: int = 0):
        def _init():
            env = MultiGPURouterEnv(
                num_gpus=num_gpus,
                max_gpu_load=10000.0,
                episode_length=100,
                verbose=False
            )
            env = Monitor(env, filename=None)
            return env
        return _init

    # Create vectorized environment
    print(f"\nCreating {n_envs} parallel environments...")
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(0)])

    # Create PPO agent
    print("\nInitializing PPO agent...")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        device=device
    )

    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)
    print(f"Policy: MlpPolicy (2-layer neural network)")
    print(f"Input: {num_gpus + 2} features [gpu0_load, ..., gpuN_load, request_length, priority]")
    print(f"Output: {num_gpus} actions [GPU 0, GPU 1, ..., GPU {num_gpus-1}]")
    print(f"Algorithm: PPO (Proximal Policy Optimization)")
    print(f"Learning rate: 3e-4")
    print(f"Device: {device}")
    print("="*70)

    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path=log_dir,
        name_prefix="multi_gpu_router_checkpoint",
        verbose=1
    )

    callbacks = [eval_callback, checkpoint_callback]

    # Train the agent
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print("Monitor progress with TensorBoard:")
    print(f"  tensorboard --logdir {log_dir}")
    print("="*70 + "\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        # Save trained model
        print(f"\n✓ Training complete!")
        print(f"Saving model to {save_path}...")
        model.save(save_path)

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Model saved: {save_path}")
        print(f"Best model: {log_dir}/best_model.zip")
        print(f"Checkpoints: {log_dir}/multi_gpu_router_checkpoint_*.zip")
        print("\nTo use the trained agent:")
        print(f"  from memopt.rl_router_env import RLRouterAgent")
        print(f"  agent = RLRouterAgent.load('{save_path}', num_gpus={num_gpus})")
        print(f"  gpu_id = agent.predict(gpu_loads=[100, 200, 150, 50], request_tokens=512)")
        print("="*70)

        # Evaluate final performance
        print("\nEvaluating final agent...")
        evaluate_agent(model, eval_env, n_eval_episodes=10)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Saving current model to {save_path}...")
        model.save(save_path)
        print("✓ Model saved.")

    finally:
        # Clean up
        env.close()
        eval_env.close()


def evaluate_agent(model, eval_env, n_eval_episodes: int = 10):
    """
    Evaluate trained agent performance.

    Args:
        model: Trained PPO model
        eval_env: Evaluation environment
        n_eval_episodes: Number of episodes to evaluate
    """
    episode_rewards = []
    episode_latencies = []
    episode_imbalances = []

    for episode in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        total_latency = 0
        total_imbalance = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)

            episode_reward += reward[0]
            total_latency += info[0].get('latency_ms', 0)
            total_imbalance += info[0].get('load_imbalance', 0)
            episode_steps += 1

        episode_rewards.append(episode_reward)
        episode_latencies.append(total_latency / episode_steps if episode_steps > 0 else 0)
        episode_imbalances.append(total_imbalance / episode_steps if episode_steps > 0 else 0)

    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Episodes: {n_eval_episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean latency: {np.mean(episode_latencies):.1f}ms ± {np.std(episode_latencies):.1f}ms")
    print(f"Mean load imbalance: {np.mean(episode_imbalances):.3f} ± {np.std(episode_imbalances):.3f}")
    print("="*70)


def main():
    import numpy as np  # Import here for evaluate_agent

    parser = argparse.ArgumentParser(
        description="Train RL agent for multi-GPU request routing"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPUs to simulate (default: 4)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps (default: 200,000)"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="multi_gpu_router.zip",
        help="Path to save trained model (default: multi_gpu_router.zip)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/multi_gpu_router",
        help="TensorBoard log directory (default: ./logs/multi_gpu_router)"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Evaluate every N steps (default: 10,000)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to train on (default: cpu)"
    )

    args = parser.parse_args()

    # Train agent
    train_multi_gpu_router(
        num_gpus=args.num_gpus,
        total_timesteps=args.timesteps,
        save_path=args.save_path,
        log_dir=args.log_dir,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        device=args.device
    )


if __name__ == "__main__":
    main()
