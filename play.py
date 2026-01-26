"""
Script de jeu pour Ping-Pong avec 3 modes:
  1. IA vs IA      : IA entrainée (gauche) vs IA entrainée (droite)
  2. Humain vs IA  : Humain (droite, souris) vs IA entrainée (gauche)
  3. Humain vs Humain : Joueur 1 (gauche, Z/Q/S/D + A/E rotation) vs Joueur 2 (droite, souris)

Usage:
    python play.py --mode ia_vs_ia              # IA vs IA (défaut)
    python play.py --mode humain_vs_ia          # Vous vs IA
    python play.py --mode humain_vs_humain      # Deux joueurs
    python play.py --model models_sb3/best/best_model.zip  # Modèle custom
    python play.py --episodes 5                 # Jouer 5 matches
    python play.py --no-render                  # Sans affichage
"""

import argparse
import os
import numpy as np
import pygame

from stable_baselines3 import PPO
from ai.environment import PingPongEnv
from config import FPS, WIDTH, HEIGHT


def load_model(model_path):
    """Charge un modèle SB3 entraîné."""
    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable: {model_path}")
        return None
    
    try:
        model = PPO.load(model_path)
        print(f"✅ Modèle chargé: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None


def get_ia_action(model, observation):
    """Récupère l'action de l'IA (déterministe)."""
    if model is None:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    action, _ = model.predict(observation, deterministic=True)
    return action


def get_human_right_action_mouse(paddle, mouse_pos, last_mouse_pos, render_mode):
    """
    Joueur DROITE : contrôle à la souris (comme dans game.py)
    - Position X et Y : suivre la souris
    - Rotation : clic gauche/droit pour rotation
    """
    mouse_x, mouse_y = mouse_pos
    
    # Positionner la raquette sous la souris
    new_x = mouse_x - paddle.width / 2
    new_y = mouse_y - paddle.height / 2
    
    # Appliquer les limites de terrain (raquette bloquée à droite)
    new_x = max(paddle.x_min, min(new_x, paddle.x_max - paddle.width))
    new_y = max(0, min(new_y, HEIGHT - paddle.height))
    
    # Calculer la vélocité basée sur le déplacement
    dt = 1.0 / FPS
    vel_x = (new_x - last_mouse_pos[0]) / dt if last_mouse_pos else 0
    vel_y = (new_y - last_mouse_pos[1]) / dt if last_mouse_pos else 0
    
    # Appliquer la nouvelle position et vélocité
    paddle.pos[0] = new_x
    paddle.pos[1] = new_y
    paddle.vel = np.array([vel_x, vel_y])
    
    # Rotation avec clics souris
    rotate = 0.0
    mouse_buttons = pygame.mouse.get_pressed()
    if mouse_buttons[0]:  # Clic gauche
        rotate = -1.0
    elif mouse_buttons[2]:  # Clic droit
        rotate = 1.0
    
    return np.array([0.0, 0.0, rotate], dtype=np.float32), (new_x, new_y)


def get_human_left_action(paddle, keys):
    """
    Joueur GAUCHE : contrôle au clavier
    - Z : Monter
    - S : Descendre
    - Q : Gauche
    - D : Droite
    - A : Rotation anti-horaire
    - E : Rotation horaire
    """
    move_x = 0.0
    move_y = 0.0
    rotate = 0.0
    
    # Mouvements en Y
    if keys[pygame.K_z]:
        move_y = -1.0
    elif keys[pygame.K_s]:
        move_y = 1.0
    
    # Mouvements en X
    if keys[pygame.K_q]:
        move_x = -1.0
    elif keys[pygame.K_d]:
        move_x = 1.0
    
    # Rotation
    if keys[pygame.K_a]:
        rotate = -1.0
    elif keys[pygame.K_e]:
        rotate = 1.0
    
    return np.array([move_x, move_y, rotate], dtype=np.float32)


def play_episode(env, model_left=None, model_right=None,
                human_left=False, human_right=False,
                use_simple_ai_right=False,
                render=True, verbose=True):
    """
    Joue un épisode avec les configurations spécifiées.
    
    Args:
        env: L'environnement Ping-Pong
        model_left: Modèle SB3 pour la raquette gauche
        model_right: Modèle SB3 pour la raquette droite
        human_left: Si True, joueur gauche est humain
        human_right: Si True, joueur droite est humain
        use_simple_ai_right: Si True, utilise l'IA simple (env._get_opponent_action) pour droite
        render: Afficher le jeu
        verbose: Afficher les infos
    
    Returns:
        dict avec statistiques de l'épisode
    """
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    steps = 0
    
    pygame.init()
    clock = pygame.time.Clock()
    
    if verbose:
        left_player = "👤 Humain" if human_left else "🤖 IA SB3" if model_left else "🤖 IA simple"
        right_player = "👤 Humain" if human_right else "🤖 IA SB3" if model_right else "🤖 IA simple"
        print(f"\n📍 Match: {left_player} (GAUCHE) vs {right_player} (DROITE)")
    
    # Tracking position souris pour calcul vélocité
    last_mouse_pos = (WIDTH // 2, HEIGHT // 2)
    
    while not done:
        # Gérer les événements Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
        
        # Récupérer l'état du clavier et de la souris
        keys = pygame.key.get_pressed()
        mouse_pos = pygame.mouse.get_pos()
        
        # === ACTION JOUEUR GAUCHE ===
        if human_left:
            action_left = get_human_left_action(env.agent_paddle if env.agent_side == "left" else env.opponent_paddle, keys)
        else:
            # IA à gauche
            if use_simple_ai_right and model_left is None:
                # IA simple pour gauche (dans IA vs IA)
                action_left = env._get_opponent_action()
            else:
                action_left = get_ia_action(model_left, obs)
        
        # === ACTION JOUEUR DROITE ===
        if human_right:
            # Humain à droite avec contrôle souris
            right_paddle = env.opponent_paddle if env.agent_side == "left" else env.agent_paddle
            action_right, last_mouse_pos = get_human_right_action_mouse(right_paddle, mouse_pos, last_mouse_pos, render)
        else:
            # IA à droite
            if use_simple_ai_right:
                # IA simple pour droite
                action_right = env._get_opponent_action()
            else:
                action_right = get_ia_action(model_right, obs)
        
        # Adapter les actions selon le côté de l'agent dans l'env
        if env.agent_side == "left":
            action = action_left  # Agent est à gauche
            opponent_action = action_right
        else:
            action = action_right  # Agent est à droite
            opponent_action = action_left
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        done = terminated or truncated
        
        # Rendu
        if render:
            env.render()
            clock.tick(FPS)
        else:
            clock.tick(FPS)
    
    # Résultats
    winner = info.get('winner_side')
    point_message = info.get('point_message', '')
    
    result = {
        'steps': steps,
        'reward': episode_reward,
        'winner': winner,
        'message': point_message,
        'left_wins': winner == 'left',
        'right_wins': winner == 'right'
    }
    
    if verbose:
        if winner == 'left':
            print(f"   ✅ Joueur GAUCHE gagne! ({point_message})")
        elif winner == 'right':
            print(f"   ✅ Joueur DROITE gagne! ({point_message})")
        else:
            print(f"   ⏸️  Match interrompu après {steps} steps")
        print(f"   📊 Reward: {episode_reward:.2f}")
    
    return result


def print_controls(mode):
    """Affiche les contrôles selon le mode."""
    print(f"\n📋 CONTRÔLES:")
    
    if mode in ['humain_vs_ia', 'humain_vs_humain']:
        print(f"   JOUEUR DROITE (Souris):")
        print(f"      🖱️  Déplacez verticalement pour contrôler la raquette")
    
    if mode == 'humain_vs_humain':
        print(f"\n   JOUEUR GAUCHE (Clavier):")
        print(f"      Z : Monter")
        print(f"      S : Descendre")
        print(f"      Q : Aller à gauche")
        print(f"      D : Aller à droite")
        print(f"      A : Rotation anti-horaire")
        print(f"      E : Rotation horaire")
    
    print(f"\n   ⌨️  ESC : Quitter le jeu")
    print(f"   {'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Jeu Ping-Pong - IA vs IA, Humain vs IA, Humain vs Humain',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python play.py --mode ia_vs_ia              # IA vs IA (défaut)
  python play.py --mode humain_vs_ia          # Vous vs IA
  python play.py --mode humain_vs_humain      # Deux joueurs
  python play.py --episodes 5                 # 5 matches
  python play.py --no-render                  # Sans affichage
        """
    )
    
    parser.add_argument('--mode', type=str, default='ia_vs_ia',
                        choices=['ia_vs_ia', 'humain_vs_ia', 'humain_vs_humain'],
                        help='Mode de jeu')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Nombre de matches à jouer')
    parser.add_argument('--model', type=str, default='models_sb3/best/best_model.zip',
                        help='Chemin vers le modèle SB3 entraîné')
    parser.add_argument('--no-render', action='store_true',
                        help='Désactiver l\'affichage')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🎮 PING-PONG - JEU INTERACTIF")
    print("="*70)
    
    # === CONFIGURATION ===
    mode = args.mode
    n_episodes = args.episodes
    render = not args.no_render
    model_path = args.model
    
    print(f"\n⚙️  Configuration:")
    print(f"   Mode: {mode}")
    print(f"   Episodes: {n_episodes}")
    print(f"   Render: {'Oui' if render else 'Non'}")
    print(f"   Modèle: {model_path}")
    
    # === CHARGER LES MODÈLES ===
    print(f"\n🤖 Chargement des modèles...")
    
    model_left = None
    model_right = None
    human_left = False
    human_right = False
    
    if mode == 'ia_vs_ia':
        print(f"   Mode: IA SB3 (GAUCHE) vs IA simple (DROITE)")
        model_left = load_model(model_path)
        if model_left is None:
            print("❌ Impossible de charger le modèle IA SB3 (gauche)")
            return
        # Pas de modèle pour droite, on utilisera l'IA simple
    
    elif mode == 'humain_vs_ia':
        print(f"   Mode: Humain (DROITE) vs IA SB3 (GAUCHE)")
        model_left = load_model(model_path)
        human_right = True
        if model_left is None:
            print("❌ Impossible de charger le modèle IA SB3 (gauche)")
            return
    
    elif mode == 'humain_vs_humain':
        print(f"   Mode: Humain (GAUCHE) vs Humain (DROITE)")
        human_left = True
        human_right = True
    
    # === CRÉER L'ENVIRONNEMENT ===
    print(f"\n🎮 Création de l'environnement...")
    render_mode = "human" if render else None
    env = PingPongEnv(render_mode=render_mode, agent_side="left", static_spawn=False)
    
    # === AFFICHER LES CONTRÔLES ===
    if human_left or human_right:
        print_controls(mode)
    
    # === BOUCLE DE JEU ===
    print(f"{'='*70}")
    print(f"🎯 Démarrage du jeu ({n_episodes} match(es))...")
    print(f"{'='*70}")
    
    stats_left = {'wins': 0, 'losses': 0}
    stats_right = {'wins': 0, 'losses': 0}
    
    try:
        for episode in range(n_episodes):
            print(f"\n🏁 Match {episode + 1}/{n_episodes}")
            
            result = play_episode(
                env,
                model_left=model_left,
                model_right=model_right,
                human_left=human_left,
                human_right=human_right,
                use_simple_ai_right=(mode == 'ia_vs_ia'),
                render=render,
                verbose=True
            )
            
            # Si l'utilisateur a quitté
            if result is None:
                print(f"\n⏸️  Jeu interrompu par l'utilisateur")
                break
            
            # Compter les victoires
            if result['left_wins']:
                stats_left['wins'] += 1
                stats_right['losses'] += 1
            elif result['right_wins']:
                stats_right['wins'] += 1
                stats_left['losses'] += 1
        
        # Résumé final
        print(f"\n{'='*70}")
        print(f"📊 STATISTIQUES FINALES ({min(episode + 1, n_episodes)} match(es))")
        print(f"{'='*70}")
        total_left = stats_left['wins'] + stats_left['losses']
        total_right = stats_right['wins'] + stats_right['losses']
        
        if total_left > 0:
            left_win_rate = stats_left['wins'] / total_left * 100
            print(f"Joueur GAUCHE: {stats_left['wins']} wins, {stats_left['losses']} losses ({left_win_rate:.1f}%)")
        else:
            print(f"Joueur GAUCHE: Pas de match joué")
        
        if total_right > 0:
            right_win_rate = stats_right['wins'] / total_right * 100
            print(f"Joueur DROITE: {stats_right['wins']} wins, {stats_right['losses']} losses ({right_win_rate:.1f}%)")
        else:
            print(f"Joueur DROITE: Pas de match joué")
        
        print(f"{'='*70}\n")
    
    except KeyboardInterrupt:
        print(f"\n⏸️  Jeu interrompu par l'utilisateur")
    
    finally:
        env.close()
        print("✅ Jeu fermé")


if __name__ == "__main__":
    main()



# Jeu / évaluation (play.py) :

# python play.py --mode ia_vs_ia --episodes 5 --model best_model.zip
# python play.py --mode humain_vs_ia --episodes 5 --model best_model.zip
# python play.py --mode humain_vs_humain --episodes 100
# Options utiles :
# --no-render (désactive l’affichage, plus rapide)
# --model <chemin.zip> (spécifie un autre modèle)
# --episodes N (nombre de matches)