"""
Script d'entraînement PPO avec Stable-Baselines3 pour Ping-Pong.

Usage:
    python train_sb3.py                      # Entraînement (100k timesteps)
    python train_sb3.py --timesteps 500000   # Timesteps personnalisés
    python train_sb3.py --load models/best_model.zip  # Continuer l'entraînement
    
Ce script utilise:
- MultiInputPolicy de SB3 pour les observations Dict
- HybridFeatureExtractor personnalisé avec embeddings spatiaux
- PPO avec les hyperparamètres optimisés pour le ping-pong
"""

import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from ai.environment import PingPongEnv
from ai.feature_extractor import HybridFeatureExtractor


class CurriculumCallback(BaseCallback):
    """
    Callback pour mettre à jour les phases de curriculum learning au fil de l'entraînement.
    Appelle env.set_episode_count() à chaque reset d'épisode.
    """
    def __init__(self, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        # Détecter les resets d'épisode via dones
        if self.locals.get("dones") is not None:
            dones = self.locals["dones"]
            if any(dones):
                self.episode_count += 1
                # Mettre à jour la phase dans l'env
                if hasattr(self.training_env, 'envs'):
                    # VecEnv
                    for env in self.training_env.envs:
                        if hasattr(env, 'set_episode_count'):
                            env.set_episode_count(self.episode_count)
                elif hasattr(self.training_env, 'set_episode_count'):
                    # Env simple
                    self.training_env.set_episode_count(self.episode_count)
                
                if self.verbose > 0 and self.episode_count % 100 == 0:
                    print(f"📊 Episode {self.episode_count}: Phase update")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Entraînement PPO avec SB3')
    parser.add_argument('--timesteps', type=int, default=100000, 
                        help='Nombre total de timesteps d\'entraînement')
    parser.add_argument('--save-freq', type=int, default=10000,
                        help='Fréquence de sauvegarde (en timesteps)')
    parser.add_argument('--load', type=str, default=None,
                        help='Chemin vers un modèle à charger pour continuer l\'entraînement')
    parser.add_argument('--render', action='store_true',
                        help='Afficher le jeu pendant l\'entraînement (ralentit beaucoup)')
    parser.add_argument('--check-env', action='store_true',
                        help='Vérifier que l\'environnement est compatible SB3')
    parser.add_argument('--embed-dim', type=int, default=16,
                        help='Dimension des embeddings spatiaux')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate pour PPO')
    parser.add_argument('--n-steps', type=int, default=2048,
                        help='Nombre de steps avant chaque update')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Taille des mini-batches')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='Nombre d\'epochs par update')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='Coefficient d\'entropie (exploration)')
    
    args = parser.parse_args()
    
    # Créer les dossiers nécessaires
    os.makedirs('models_sb3', exist_ok=True)
    os.makedirs('logs_sb3', exist_ok=True)
    
    # === CRÉER L'ENVIRONNEMENT ===
    print("="*70)
    print("🎮 Création de l'environnement Ping-Pong")
    print("="*70)
    
    render_mode = "human" if args.render else None
    env = PingPongEnv(render_mode=render_mode, agent_side="left", static_spawn=False)
    
    # Wrapper Monitor pour logging automatique
    env = Monitor(env, 'logs_sb3')
    
    # === VÉRIFICATION DE L'ENVIRONNEMENT (optionnel) ===
    if args.check_env:
        print("\n🔍 Vérification de la compatibilité SB3...")
        try:
            check_env(env, warn=True)
            print("✅ Environnement compatible avec SB3 !")
        except Exception as e:
            print(f"❌ Erreur de compatibilité: {e}")
            return
    
    # === CONFIGURATION DE LA POLICY ===
    print("\n⚙️  Configuration de la policy PPO")
    print("="*70)
    
    policy_kwargs = dict(
        # Notre custom feature extractor avec embeddings
        features_extractor_class=HybridFeatureExtractor,
        features_extractor_kwargs=dict(embed_dim=args.embed_dim),
        
        # Architecture des réseaux Actor/Critic après le feature extractor
        # net_arch: couches partagées puis séparées pour pi (policy) et vf (value function)
        net_arch=dict(
            pi=[256, 256],  # 2 couches de 256 neurones pour l'actor
            vf=[256, 256]   # 2 couches de 256 neurones pour le critic
        ),
        
        # Fonction d'activation
        activation_fn=th.nn.ReLU,
    )
    
    print(f"  • Feature Extractor: HybridFeatureExtractor (embed_dim={args.embed_dim})")
    print(f"  • Architecture: pi=[256, 256], vf=[256, 256]")
    print(f"  • Learning rate: {args.learning_rate}")
    print(f"  • n_steps: {args.n_steps}")
    print(f"  • batch_size: {args.batch_size}")
    print(f"  • n_epochs: {args.n_epochs}")
    print(f"  • gamma: {args.gamma}")
    print(f"  • gae_lambda: {args.gae_lambda}")
    print(f"  • ent_coef: {args.ent_coef}")
    
    # === CRÉER OU CHARGER LE MODÈLE ===
    print("\n🤖 Initialisation du modèle PPO")
    print("="*70)
    
    if args.load and os.path.exists(args.load):
        print(f"📂 Chargement du modèle depuis: {args.load}")
        model = PPO.load(
            args.load,
            env=env,
            custom_objects={
                "learning_rate": args.learning_rate,
                "policy_kwargs": policy_kwargs
            }
        )
        print("✅ Modèle chargé avec succès!")
    else:
        if args.load:
            print(f"⚠️  Fichier {args.load} introuvable, création d'un nouveau modèle")
        
        model = PPO(
            "MultiInputPolicy",  # OBLIGATOIRE pour Dict observation space
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="./logs_sb3/tensorboard/"
        )
        print("✅ Nouveau modèle créé!")
    
    # === CALLBACKS ===
    # Curriculum learning: met à jour les phases d'entraînement
    curriculum_callback = CurriculumCallback(verbose=1)
    
    # Sauvegarde périodique
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path='./models_sb3/checkpoints/',
        name_prefix='ppo_pingpong',
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # Environnement d'évaluation (sans render)
    eval_env = Monitor(PingPongEnv(render_mode=None, agent_side="left", static_spawn=False), 
                       'logs_sb3/eval')
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models_sb3/best/',
        log_path='./logs_sb3/eval/',
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )
    
    callbacks = [curriculum_callback, checkpoint_callback, eval_callback]
    
    # === ENTRAÎNEMENT ===
    print("\n🚀 Début de l'entraînement")
    print("="*70)
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Sauvegarde tous les {args.save_freq:,} timesteps")
    print(f"TensorBoard: logs_sb3/tensorboard/")
    print("="*70)
    print("\n💡 Astuce: Lancez TensorBoard pour suivre l'entraînement en temps réel:")
    print("   tensorboard --logdir=./logs_sb3/tensorboard/\n")
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Entraînement interrompu par l'utilisateur")
    
    # === SAUVEGARDE FINALE ===
    print("\n💾 Sauvegarde du modèle final...")
    final_path = './models_sb3/ppo_pingpong_final.zip'
    model.save(final_path)
    print(f"✅ Modèle sauvegardé: {final_path}")
    
    print("\n" + "="*70)
    print("🎉 Entraînement terminé!")
    print("="*70)
    print(f"📁 Modèle final: {final_path}")
    print(f"📁 Meilleur modèle: ./models_sb3/best/best_model.zip")
    print(f"📁 Checkpoints: ./models_sb3/checkpoints/")
    print(f"📊 Logs: ./logs_sb3/")
    print("="*70)
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    # Import torch ici pour éviter l'erreur si pas utilisé ailleurs
    import torch as th
    main()


# pour continuer l'entraînement à partir du modèle final:
# python train_sb3.py --load models_sb3/ppo_pingpong_final.zip --timesteps 100000

# Continuer l'entraînement à partir du meilleur modèle:
# python train_sb3.py --load models_sb3/best/best_model.zip --timesteps 100000

# checkpoint spécifique:
# python train_sb3.py --load models_sb3/checkpoints/ppo_pingpong_50000_steps.zip --timesteps 100000