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
    Callback pour :
    - suivre le numéro d'épisode (persistant entre relances d'entraînement)
    - mettre à jour le curriculum via env.set_episode_count()
    - sauvegarder le numéro d'épisode dans episode_count.txt
    - logger le numéro d'épisode dans TensorBoard (train/episode_count)
    """

    def __init__(self, start_episode_count: int = 0, monitor_dir: str = None, verbose: int = 0):
        super(CurriculumCallback, self).__init__(verbose)
        self.episode_count = start_episode_count
        self.monitor_dir = monitor_dir

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

                # Sauvegarder dans episode_count.txt
                if self.monitor_dir:
                    _save_episode_count(self.monitor_dir, self.episode_count)

                # Log TensorBoard
                if self.model is not None and self.model.logger is not None:
                    self.model.logger.record("train/episode_count", float(self.episode_count))

                if self.verbose > 0 and self.episode_count % 100 == 0:
                    print(f"📊 Episode {self.episode_count}: Phase update")

        return True


def _detect_log_name_from_path(model_path: str) -> str:
    """Détecte le log_name depuis le chemin du modèle chargé.
    
    Gère 3 cas:
    1. models_sb3/best/[log_name]/fichier.zip
    2. models_sb3/checkpoints/[log_name]/fichier.zip
    3. models_sb3/ppo_pingpong_[log_name]_final.zip
    """
    if not model_path or not os.path.exists(model_path):
        return None
    
    normalized = model_path.replace('\\', '/')
    
    # Cas 1 & 2: best/ ou checkpoints/
    for folder in ['/best/', '/checkpoints/']:
        if folder in normalized:
            parts = normalized.split(folder)
            if len(parts) > 1:
                # Extraire le nom du dossier après best/ ou checkpoints/
                log_name = parts[1].split('/')[0]
                if log_name and log_name.endswith('.zip'):
                    # Si c'est directement le fichier, pas de sous-dossier
                    continue
                return log_name
    
    # Cas 3: ppo_pingpong_[log_name]_final.zip
    if 'ppo_pingpong_' in normalized and '_final.zip' in normalized:
        start = normalized.index('ppo_pingpong_') + len('ppo_pingpong_')
        end = normalized.index('_final.zip')
        return normalized[start:end]
    
    return None


def _load_episode_count(monitor_dir: str) -> int:
    """Retourne le nombre d'épisodes déjà terminés en lisant episode_count.txt.

    - Si le fichier n'existe pas → 0
    - Le fichier contient un seul nombre entier
    """
    episode_count_file = os.path.join(monitor_dir, "episode_count.txt")
    if not os.path.isfile(episode_count_file):
        return 0

    try:
        with open(episode_count_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content.isdigit():
            return int(content)
        return 0
    except Exception:
        # En cas de souci de lecture, ne pas bloquer l'entraînement
        return 0


def _save_episode_count(monitor_dir: str, episode_count: int) -> None:
    """Sauvegarde le numéro d'épisode dans episode_count.txt."""
    try:
        episode_count_file = os.path.join(monitor_dir, "episode_count.txt")
        with open(episode_count_file, "w", encoding="utf-8") as f:
            f.write(str(episode_count))
    except Exception as e:
        print(f"⚠️  Erreur lors de la sauvegarde de episode_count.txt : {e}")


def _unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def _apply_to_base_envs(env, fn):
    if hasattr(env, "envs"):
        for sub_env in env.envs:
            fn(_unwrap_env(sub_env))
    else:
        fn(_unwrap_env(env))


def _set_self_play(env, model, deterministic=False):
    # On définit la logique directement pour chaque instance d'environnement
    def set_on_env(base_env):
        # Cette fonction capture 'base_env' et sera appelée à chaque step par l'environnement
        def opponent_policy(obs):
            # 1. Prédiction (Le modèle pense toujours jouer à GAUCHE)
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # 2. Inversion DYNAMIQUE selon le côté de l'adversaire
            # Si l'agent est à GAUCHE, l'adversaire est à DROITE -> Il faut inverser l'action (Miroir)
            # Si l'agent est à DROITE, l'adversaire est à GAUCHE -> L'action est déjà bonne (Directe)
            if base_env.agent_side == "left":
                # L'adversaire est à droite : on inverse X et la Rotation
                opp_action = action.copy()
                opp_action[0] = -opp_action[0]  # Inversion X (Gauche <-> Droite)
                opp_action[2] = -opp_action[2]  # Inversion Rotation (Horaire <-> Anti-horaire)
                return opp_action
            else:
                # L'adversaire est à gauche : on applique l'action telle quelle
                return action

        # On injecte cette policy intelligente dans l'environnement
        if hasattr(base_env, "set_opponent_policy"):
            base_env.set_opponent_policy(opponent_policy)

    _apply_to_base_envs(env, set_on_env)


def main():
    parser = argparse.ArgumentParser(description='Entraînement PPO avec SB3')
    parser.add_argument('--timesteps', type=int, default=100000, 
                        help='Nombre total de timesteps d\'entraînement')
    parser.add_argument('--save-freq', type=int, default=20000,
                        help='Fréquence de sauvegarde (en timesteps)')
    parser.add_argument('--load', type=str, default=None,
                        help='Chemin vers un modèle à charger pour continuer l\'entraînement')
    parser.add_argument('--render', action='store_true',
                        help='Afficher le jeu pendant l\'entraînement (ralentit beaucoup)')
    parser.add_argument('--check-env', action='store_true',
                        help='Vérifier que l\'environnement est compatible SB3')
    parser.add_argument('--embed-dim', type=int, default=16,
                        help='Dimension des embeddings spatiaux')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                        help='Learning rate pour PPO')
    parser.add_argument('--n-steps', type=int, default=2048,
                        help='Nombre de steps avant chaque update')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Taille des mini-batches')
    parser.add_argument('--n-epochs', type=int, default=5,
                        help='Nombre d\'epochs par update')
    parser.add_argument('--gamma', type=float, default=0.995,
                        help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--ent-coef', type=float, default=0.005,
                        help='Coefficient d\'entropie (exploration)')
    parser.add_argument('--log-name', type=str, default=None,
                        help='Nom de la run TensorBoard (ignoré si on charge un modèle)')
    parser.add_argument('--random-side', action='store_true',
                        help='Alterner aléatoirement le côté agent à chaque épisode')
    parser.add_argument('--self-play', action='store_true',
                        help='Utiliser la policy courante pour l\'adversaire (self-play)')
    parser.add_argument('--self-play-deterministic', action='store_true',
                        help='Forcer l\'adversaire en mode deterministic')
    
    args = parser.parse_args()
    
    # === DÉTERMINER LE LOG_NAME ===
    # Si on charge un modèle, extraire son log_name original du chemin
    if args.load:
        detected_log_name = _detect_log_name_from_path(args.load)
        if detected_log_name:
            args.log_name = detected_log_name
            print(f"🔍 Log_name détecté du modèle chargé: {detected_log_name}")
        elif not os.path.exists(args.load):
            print(f"⚠️  Fichier {args.load} introuvable!")
    
    # Créer les dossiers nécessaires
    os.makedirs('models_sb3', exist_ok=True)
    os.makedirs('logs_sb3', exist_ok=True)
    
    # === DÉTERMINER LES CHEMINS (APRÈS détection du log_name) ===
    if args.log_name:
        tensorboard_log = f"./logs_sb3/tensorboard/{args.log_name}/"
        checkpoint_path = f"./models_sb3/checkpoints/{args.log_name}/"
        best_model_path = f"./models_sb3/best/{args.log_name}/"
        monitor_log_path = f"./logs_sb3/{args.log_name}/"
        eval_log_path = f"./logs_sb3/{args.log_name}/eval/"
        final_model_name = f"ppo_pingpong_{args.log_name}_final.zip"
        print(f"📊 Run TensorBoard: {args.log_name}")
        print(f"💾 Checkpoints: {checkpoint_path}")
        print(f"🏆 Best model: {best_model_path}")
    else:
        tensorboard_log = "./logs_sb3/tensorboard/"
        checkpoint_path = "./models_sb3/checkpoints/"
        best_model_path = "./models_sb3/best/"
        monitor_log_path = "./logs_sb3/"
        eval_log_path = "./logs_sb3/eval/"
        final_model_name = "ppo_pingpong_final.zip"
        print(f"📊 Run TensorBoard: auto-numbered (PPO_1, PPO_2, ...)")
    
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(best_model_path, exist_ok=True)
    os.makedirs(monitor_log_path, exist_ok=True)
    os.makedirs(eval_log_path, exist_ok=True)
    
    # === CRÉER L'ENVIRONNEMENT ===
    print("="*70)
    print("Création de l'environnement Ping-Pong")
    print("="*70)

    # Reprendre le numéro d'épisode si on continue la même run (monitor.csv existant)
    start_episode_count = _load_episode_count(monitor_log_path)
    print(f"ℹ️  Monitor utilisé : {monitor_log_path}")
    print(f"ℹ️  Épisodes déjà terminés (monitor.csv) : {start_episode_count}")
    if start_episode_count > 0:
        print(f"🔄 Reprise à l'épisode {start_episode_count}")


    import shutil
    monitor_file = os.path.join(monitor_log_path, "monitor.csv")
    backup_monitor_file = None
    # Sauvegarder l'ancien monitor.csv s'il existe
    if os.path.isfile(monitor_file):
        backup_monitor_file = monitor_file + ".bak"
        shutil.copyfile(monitor_file, backup_monitor_file)

    render_mode = "human" if args.render else None
    env = PingPongEnv(render_mode=render_mode, agent_side="left", static_spawn=False, game_mode=False, randomize_agent_side=args.random_side)

    # Initialiser l'environnement avec le bon compteur d'épisodes dès le début
    if start_episode_count > 0:
        env.set_episode_count(start_episode_count)
        print(f"✅ Phase de curriculum initialisée: Phase {env.training_phase}")

    # Wrapper Monitor pour logging automatique (SB3 va écraser monitor.csv)
    env = Monitor(env, monitor_log_path)

    # Fusionner l'ancien monitor.csv avec le nouveau si besoin
    if backup_monitor_file and os.path.isfile(monitor_file):
        try:
            with open(backup_monitor_file, "r", encoding="utf-8") as f_old:
                old_lines = f_old.readlines()
            with open(monitor_file, "r", encoding="utf-8") as f_new:
                new_lines = f_new.readlines()
            # En-tête = 2 lignes, puis données
            merged_lines = old_lines[:2]
            # Ajoute toutes les anciennes données (sauf en-tête)
            merged_lines += [l for l in old_lines[2:] if l.strip()]
            # Ajoute les nouvelles données (sauf en-tête)
            merged_lines += [l for l in new_lines[2:] if l.strip()]
            # Écrit le fichier fusionné
            with open(monitor_file, "w", encoding="utf-8") as f_out:
                f_out.writelines(merged_lines)
            os.remove(backup_monitor_file)
            print(f"✅ Fusion de l'historique monitor.csv terminée.")
        except Exception as e:
            print(f"⚠️  Erreur lors de la fusion de monitor.csv : {e}")
    
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
        features_extractor_kwargs=dict(embed_dim=args.embed_dim), # les arguments de la class au dessus
        
        # Architecture des réseaux Actor/Critic après le feature extractor
        # net_arch: couches partagées puis séparées pour pi (policy) et vf (value function)
        net_arch=dict(
            pi=[512, 256],  # 2 couches de 256 neurones pour l'actor
            vf=[512, 256]   # 2 couches de 256 neurones pour le critic
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
    print(f"  • clip_range: 0.3")
    print(f"  • ent_coef: {args.ent_coef}")
    
    # === CRÉER OU CHARGER LE MODÈLE ===
    print("\nInitialisation du modèle PPO")
    print("="*70)
    
    from stable_baselines3.common.logger import configure

    if args.load and os.path.exists(args.load):
        print(f"📂 Chargement du modèle depuis: {args.load}")
        if args.log_name:
            print(f"⚠️  --log-name ignoré (on continue avec la run existante)")
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
            clip_range=0.3,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=tensorboard_log
        )
        print("✅ Nouveau modèle créé!")

    # Forcer l'écriture des logs tensorboard dans le dossier sans PPO_0
    new_logger = configure(tensorboard_log, ["tensorboard"])
    model.set_logger(new_logger)
    
    # === CALLBACKS ===
    # Curriculum learning: met à jour les phases d'entraînement
    curriculum_callback = CurriculumCallback(start_episode_count=start_episode_count, monitor_dir=monitor_log_path, verbose=1)
    
    # Sauvegarde périodique
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=checkpoint_path,
        name_prefix='ppo_pingpong',
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # Environnement d'évaluation (sans render) - Toujours en phase finale (game_mode=True)
    eval_env = Monitor(PingPongEnv(render_mode=None, agent_side="left", static_spawn=False, game_mode=False, randomize_agent_side=args.random_side), 
                       eval_log_path)
    
    if args.self_play:
        print("✅ Self-play activé sur l'environnement d'évaluation")
        # On force le mode déterministe pour l'évaluation pour des résultats stables
        _set_self_play(eval_env, model, deterministic=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=eval_log_path,
        eval_freq=100000,
        deterministic=True,
        render=False,
        n_eval_episodes=30,
    )
    
    callbacks = [curriculum_callback, checkpoint_callback, eval_callback]
    
    # === SELF-PLAY (optionnel) ===
    if args.self_play:
        _set_self_play(model.get_env(), model, deterministic=args.self_play_deterministic)
        print("✅ Self-play activé : l'adversaire utilise la policy courante")

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
            progress_bar=True,
            reset_num_timesteps=False  # Continuer la même run TensorBoard si on charge un modèle
        )
    except KeyboardInterrupt:
        print("\n⚠️  Entraînement interrompu par l'utilisateur")
    
    # === SAUVEGARDE FINALE ===
    print("\n💾 Sauvegarde du modèle final...")
    final_path = f'./models_sb3/{final_model_name}'
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

# Creer un nouveau modèle:
# python train_sb3.py --log-name modele_1 --timesteps 200000 --random-side --self-play

# Continuer l'entraînement à partir du modèle final:
# python train_sb3.py --load models_sb3/ppo_pingpong_modele_1_final.zip --timesteps 1000000 --render --random-side --self-play

# Continuer l'entraînement à partir du meilleur modèle:
# python train_sb3.py --load models_sb3/best/best_model.zip --timesteps 100000 --random-side --self-play

# checkpoint spécifique:
# python train_sb3.py --load models_sb3/checkpoints/ppo_pingpong_50000_steps.zip --timesteps 100000 --random-side --self-play

# tensorboard:
# tensorboard --logdir=./logs_sb3/tensorboard/