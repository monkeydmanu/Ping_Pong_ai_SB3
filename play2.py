"""
Script d'entraînement PPO pour Ping-Pong.
Style Phil's code - simple et efficace.

Usage:
    python train.py                      # Entraînement (1000 épisodes)
    python train.py --render             # Avec affichage
    python train.py --mode play          # Jouer avec un modèle entraîné
    python train.py --episodes 500       # Nombre d'épisodes personnalisé
"""

import os
import argparse
import numpy as np
import pickle
from config import FPS
from stable_baselines3 import PPO

from ai.environment import PingPongEnv

# Pygame sera importé seulement si nécessaire
try:
    import pygame
except ImportError:
    pygame = None

DEFAULT_MODEL_PATH = "models_sb3/best/best_model.zip"



def load_training_state(checkpoint_dir):
    """Charge l'état complet de l'entraînement."""
    checkpoint_file = os.path.join(checkpoint_dir, 'training_state.pkl')
    
    if not os.path.exists(checkpoint_file):
        print(f"⚠️  Aucun état d'entraînement trouvé: {checkpoint_file}")
        return None
    
    try:
        with open(checkpoint_file, 'rb') as f:
            state = pickle.load(f)
        print(f"✅ État d'entraînement chargé: {checkpoint_file}")
        print(f"   Reprise à partir de l'épisode {state['episode'] + 1}")
        return state
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'état: {e}")
        return None



def load_best_model(model_path: str = DEFAULT_MODEL_PATH):
    """Charge un modèle SB3 (.zip)."""
    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable: {model_path}")
        return None

    try:
        model = PPO.load(model_path)
        print(f"✅ Modèle chargé: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None


def predict_action(model, observation, deterministic: bool = True):
    """Prédit une action continue [-1,1]^3 avec le modèle SB3."""
    if model is None:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    action, _ = model.predict(observation, deterministic=deterministic)
    return action




def _update_ball_debug_info(game):
    """Met à jour les infos de debug pour la vitesse et le spin de la balle."""
    if game.env.ball_in_play and game.env.ball:
        game.last_ball_vel = (game.env.ball.vel[0], game.env.ball.vel[1])
        game.last_spin = game.env.ball.angular_speed


def play_ai_vs_ai(model_path: str = DEFAULT_MODEL_PATH, num_episodes: int = 5, vs_trained: bool = False):
    """IA vs IA avec affichage visuel (évaluation sans entraînement)."""
    # Importer Game uniquement quand nécessaire
    from engine.game import Game

    model_left = load_best_model(model_path)
    model_right = load_best_model(model_path) if vs_trained else None

    if model_left is None:
        print("❌ Impossible de charger le modèle IA (gauche)")
        return
    if vs_trained and model_right is None:
        print("❌ Impossible de charger le modèle IA (droite)")
        return

    print("=== Mode Jeu IA vs IA ===")
    if vs_trained:
        print("IA gauche: Modèle entraîné")
        print("IA droite: Modèle entraîné")
    else:
        print("IA gauche: Modèle entraîné")
        print("IA droite: IA simple basique")

    game = Game(player1_type="ai", player2_type="ai")

    for episode in range(num_episodes):
        obs, _ = game.env.reset()
        game.score_left = 0
        game.score_right = 0

        steps = 0
        done = False

        while not done and game.running and steps < 3000:
            # Gestion des events pour garder la fenêtre réactive (fermeture possible)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    game.running = False
                    break

            if not game.running:
                break

            # IA gauche (agent_paddle) - utilise le modèle entraîné
            action_p1 = predict_action(model_left, obs, deterministic=True)

            # IA droite (opponent_paddle)
            if vs_trained and model_right is not None:
                action_p2 = predict_action(model_right, obs, deterministic=True)
            else:
                action_p2 = game.env._get_opponent_action()

            # Simuler directement via env (pas de reward car pas d'entraînement)
            obs, reward, terminated, truncated, info = game.env.step(action_p1, action_p2)
            done = terminated or truncated
            game.score_left = info.get('score_left', 0)
            game.score_right = info.get('score_right', 0)
            game.point_message = info.get('point_message', '')

            steps += 1

            # Mettre à jour les infos de debug (vitesse et spin de la balle)
            _update_ball_debug_info(game)

            # Afficher visuellement
            game.draw()
            game.clock.tick(FPS)

        if not game.running:
            break

        print(f"Episode {episode + 1}: {game.score_left} - {game.score_right}")

    game.running = False
    pygame.quit()


def play_human_vs_human(mouse: bool = False, mouse_side: str = 'right'):
    """
    1v1 entre deux joueurs humains avec affichage complet.
    Joueur 1 (gauche): Souris si mouse_control_p1=True, sinon Z/S (vertical), Q/D (horizontal), A/E (rotation)
    Joueur 2 (droite): O/L (vertical), K/M (horizontal), I/P (rotation)
    """
    from engine.game import Game
    
    print("=== Mode Humain vs Humain ===")
    p1_mouse = mouse and mouse_side == 'left'
    p2_mouse = mouse and mouse_side == 'right'
    if p1_mouse:
        print("Joueur 1 (gauche): SOURIS (clics pour rotation)")
    else:
        print("Joueur 1 (gauche): Z/S=vertical, Q/D=horizontal, A/E=rotation")
    if p2_mouse:
        print("Joueur 2 (droite): SOURIS (clics pour rotation)")
    else:
        print("Joueur 2 (droite): O/L=vertical, K/M=horizontal, I/P=rotation")
    
    game = Game(player1_type="human", player2_type="human", mouse_control_p1=p1_mouse, mouse_control_p2=p2_mouse)
    game.run()
    
    print("Fin du jeu!")


def play_ai_vs_human(model_path: str = DEFAULT_MODEL_PATH, mouse: bool = False, mouse_side: str = 'right'):
    """IA vs Joueur Humain avec affichage complet."""
    from engine.game import Game

    model_ai = load_best_model(model_path)
    if model_ai is None:
        print("❌ Impossible de charger le modèle IA")
        return

    human_on_right = mouse_side == 'right'

    print("=== Mode IA vs Humain ===")
    if human_on_right:
        # Humain à droite (joueur 2) à la souris
        print("Vous êtes le joueur 2 (droite): SOURIS (clics pour rotation)") if mouse else print("Vous êtes le joueur 2 (droite): O/L=vertical, K/M=horizontal, I/P=rotation")
        print("L'IA est le joueur 1 (gauche)")
        game = Game(player1_type="ai", player2_type="human", mouse_control_p2=mouse, agent_side="left")
    else:
        # Humain à gauche (joueur 1)
        print("Vous êtes le joueur 1 (gauche): SOURIS (clics pour rotation)") if mouse else print("Vous êtes le joueur 1 (gauche): Z/S=vertical, Q/D=horizontal, A/E=rotation")
        print("L'IA est le joueur 2 (droite)")
        game = Game(player1_type="human", player2_type="ai", mouse_control_p1=mouse, agent_side="right")

    obs, _ = game.env.reset()

    # Boucle de jeu custom (on fournit l'action IA pour le joueur contrôlé par l'IA)
    while game.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.running = False
        if not game.running:
            break

        game.handle_events()
        if not game.running:
            break

        if human_on_right:
            # IA contrôle joueur 1 (gauche) => action_p1
            action_p1 = predict_action(model_ai, obs, deterministic=True)
            action_p2 = game.action_p2  # humain droite
        else:
            # IA contrôle joueur 2 (droite) => action_p2
            action_p1 = game.action_p1  # humain gauche
            action_p2 = predict_action(model_ai, obs, deterministic=True)

        # affichage vel joueur gauche avant step
        #print(f"Vel joueur gauche avant step: {game.env.agent_paddle.vel}")

        # affichage vel joueur droite avant step
        #print(f"Vel joueur droite avant step: {game.env.opponent_paddle.vel}")

        obs, reward, terminated, truncated, info = game.env.step(action_p1, action_p2)
        done = terminated or truncated
        game.score_left = info.get('score_left', 0)
        game.score_right = info.get('score_right', 0)

        # affichage vel joueur gauche après step
        #print(f"Vel joueur gauche après step: {game.env.agent_paddle.vel}")

        # affichage vel joueur droite après step
        #print(f"Vel joueur droite après step: {game.env.opponent_paddle.vel}\n")

        if done:
            game.point_message = info.get('point_message', '')
            game.message_timer = 120
            obs, _ = game.env.reset()

        if game.message_timer > 0:
            game.message_timer -= 1

        _update_ball_debug_info(game)

        game.draw()
        game.clock.tick(FPS)

    pygame.quit()
    print("Fin du jeu!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ping-Pong SB3 - Modes de jeu')
    parser.add_argument('--mode', type=str, default='play',
                        choices=['play', 'human', 'ai_vs_human'],
                        help='play: IA vs IA, human: humain vs humain, ai_vs_human: IA vs humain')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Nombre d\'épisodes / matches')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Chemin vers le meilleur modèle .zip SB3')
    parser.add_argument('--vs-trained', action='store_true',
                        help='En mode play (IA vs IA), utiliser le modèle entraîné pour les deux côtés')
    parser.add_argument('--mouse', action='store_true',
                        help='Activer le contrôle à la souris pour un joueur')
    parser.add_argument('--mouse-side', type=str, choices=['left', 'right'], default='right',
                        help='Choisir quel joueur utilise la souris (left ou right). Par défaut: right')
    
    args = parser.parse_args()
    
    if args.mode == 'play':
        play_ai_vs_ai(model_path=args.model_path, num_episodes=args.episodes, vs_trained=args.vs_trained)
    elif args.mode == 'human':
        play_human_vs_human(mouse=args.mouse, mouse_side=args.mouse_side)
    elif args.mode == 'ai_vs_human':
        play_ai_vs_human(model_path=args.model_path, mouse=args.mouse, mouse_side=args.mouse_side)




# # IA vs Humain, humain à droite à la souris
# python play2.py --mode ai_vs_human --mouse --mouse-side right --model_path models_sb3/best/modele_1/best_model.zip

# # IA vs Humain, humain à gauche à la souris
# python play2.py --mode ai_vs_human --mouse --mouse-side left --model_path models_sb3/best/modele_1/best_model.zip

# # Humain vs Humain, droite à la souris
# python play2.py --mode human --mouse --mouse-side right