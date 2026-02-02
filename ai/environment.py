"""
Environnement Gymnasium pour le Ping-Pong.
Compatible avec Stable-Baselines3 pour l'entraînement PPO.
"""

import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from config import (
    WIDTH, HEIGHT, FPS, TABLE_Y, PIXELS_PER_METER,
    RACKET_WIDTH_PX, RACKET_HEIGHT_PX, TABLE_WIDTH_PX,
    ADAPTIVE_BOUNDARY_OFFSET, OUT_MARGIN
)
from core.ball import Ball, spawn_ball_left, spawn_ball_right
from core.paddle import Paddle
from core.net import Net
from core.table import Table
from engine.collision import check_ball_paddle, check_ball_net, check_table_collision

# Limite de steps par épisode pour éviter les boucles infinies
MAX_STEPS_PER_EPISODE = 500


class PingPongEnv(gym.Env):
    """
    Environnement Ping-Pong pour reinforcement learning.
    
    Observation (Format Dict - MultiInputPolicy):
        Embeddings (index discrets):
            - ball_idx: Position balle en grille 16x16 [0-255]
            - paddle_idx: Position raquette agent en grille 16x16 [0-255]
            - angle_idx: Angle raquette en 16 bins [0-15]
        
        Continuous (18 valeurs normalisées [-1, 1]):
            [0-1]   Vitesse balle (vx, vy)
            [2]     Spin balle
            [3-4]   Vitesse raquette agent (vx, vy)
            [5]     Balle de notre côté ? (1=oui, -1=non)
            [6]     Balle vient vers nous ? (1=oui, -1=non)
            [7]     Rebonds sur notre côté (0, 0.5, 1)
            [8]     Rebonds côté adverse (0, 0.5, 1)
            [9]     Est-ce un service ? (1=oui, -1=non)
            [10-11] Distance balle→raquette (dx, dy)
            [12-13] Position raquette agent (x, y)
            [14-15] Vitesse relative balle-raquette (pour timing)
            [16-17] sin(angle) et cos(angle) (continuité circulaire)
    
    Actions (3 valeurs continues [-1, 1]):
        - move_x : mouvement horizontal
        - move_y : mouvement vertical
        - rotate : rotation de la raquette
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}
    
    def __init__(self, render_mode=None, agent_side="left", player1_mouse_control=False, player2_mouse_control=False, static_spawn=False, game_mode=False):
        """
        Initialise l'environnement Ping-Pong.
        
        Args:
            render_mode (str): Mode de rendu ("human" ou "rgb_array")
            agent_side (str): Côté de l'agent ("left" ou "right")
            player1_mouse_control (bool): Si True, le joueur 1 est contrôlé à la souris
                (dans ce cas, on n'applique que la rotation, pas le mouvement)
            player2_mouse_control (bool): Si True, le joueur 2 est contrôlé à la souris
            static_spawn (bool): Si True, la balle apparaît immobile sans gravité jusqu'au premier contact raquette.
            game_mode (bool): Si True, ignore le curriculum et place les raquettes à des positions fixes pour le jeu (pas l'entraînement).
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.agent_side = agent_side  # "left" ou "right"
        self.static_spawn = static_spawn

        # Mode jeu (non-entrainement)
        self.game_mode = game_mode
        
        # === ESPACE D'OBSERVATION (SB3 MultiInputPolicy) ===
        # Format Dict avec embeddings spatiaux pour efficacité
        # Les index sont mappés à une grille 16x16 (256 cellules)
        self.observation_space = spaces.Dict({
            # Index de cellule pour la position de la balle (grille 16x16)
            "ball_idx": spaces.Box(low=0, high=255, shape=(1,), dtype=np.int64),
            # Index de cellule pour la position de la raquette agent
            "paddle_idx": spaces.Box(low=0, high=255, shape=(1,), dtype=np.int64),
            # Index d'angle discret pour la raquette (16 bins, 22.5° chacun)
            "angle_idx": spaces.Box(low=0, high=15, shape=(1,), dtype=np.int64),
            # Features continues normalisées dans [-1, 1]
            # [0-1] Vitesse balle (vx, vy)
            # [2] Spin balle
            # [3-4] Vitesse raquette agent (vx, vy)
            # [5] Balle de notre côté ? (1/-1)
            # [6] Balle vient vers nous ? (1/-1)
            # [7] Rebonds notre côté (0/0.5/1)
            # [8] Rebonds côté adverse (0/0.5/1)
            # [9] Service ? (1/-1)
            # [10-11] Distance balle→raquette (dx, dy)
            # [12-13] Position raquette agent (x, y)
            # [14-15] Vitesse relative balle-raquette (vx_rel, vy_rel) - NOUVEAU pour timing
            # [16-17] sin(angle) et cos(angle) - NOUVEAU pour continuité circulaire
            # [18] Ai-je le droit de frapper ? (1=oui après rebond, -1=non avant rebond) - CRITIQUE
            "continuous": spaces.Box(low=-1.0, high=1.0, shape=(19,), dtype=np.float32)
        })
        
        # === ESPACE D'ACTION (Continu) ===
        # [move_x, move_y, rotate] dans [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # === Contrôle du joueur 1 et 2 ===
        self.player1_mouse_control = player1_mouse_control
        self.player2_mouse_control = player2_mouse_control
        
        # Initialisation Pygame (optionnel pour le rendu)
        self.screen = None
        self.clock = None
        
        # Objets du jeu
        self.table = None
        self.net = None
        self.ball = None
        self.agent_paddle = None
        self.opponent_paddle = None
        
        # État du jeu
        self.steps = 0
        self.last_hit_by = None  # "agent" ou "opponent"
        self.ball_in_play = False
        self.ball_side = None  # 'left' ou 'right' - côté actuel de la balle
        self.is_agent_service = True
        
        # Scores
        self.score_left = 0
        self.score_right = 0
        self.last_point_message = ""
        
        # Flags pour les récompenses (éviter les doublons)
        self.bounce_reward_given = False
        self.proximity_reward_given = False
        self.fault_volley = False
        self.opponent_fault_volley = False
        self.double_hit_fault = False
        self.service_fault = False
        self.pending_hit_reward = False
        self.pending_wrong_direction = False  # Frappe dans la mauvaise direction (pending)
        self.ball_out_result = None  # 'win' ou 'loss'
        self.point_winner_side = None  # 'left' ou 'right'
        
        # Tracking pour reward shaping (distance à la balle)
        self.prev_dist_to_ball = None
        
        # Accumulation des rewards pour l'épisode
        self.episode_reward_accumulation = 0.0

        # Proximité du passage de balle à TABLE_Y (après frappe agent)
        self.table_cross_proximity = None
        
        # === CURRICULUM LEARNING ===
        # Phase 0: Balle lente proche (0-200 épisodes)
        # Phase 1: Balle modérée (201-600)
        # Phase 2: Balle normale (601-1200)
        # Phase 3: Adversaire compétent (1201+)
        self.training_phase = 0
        self.episode_count = 0
        self.opponent_difficulty = 0.0  # 0=passif, 1=compétent
        self._speed_factor = 1.0  # Facteur appliqué à chaque update (curriculum)
        
    def set_episode_count(self, count):
        """Met à jour le compteur d'épisodes et ajuste la phase en conséquence."""
        self.episode_count = count
        # Phase automatique basée sur l'épisode (curriculum rallongé)
        if count < 2000:
            self.training_phase = 0  # Phase d'apprentissage de base (2000 épisodes)
            self.opponent_difficulty = 0.0
        elif count < 4000:
            self.training_phase = 0  # Phase intermédiaire (2000 épisodes)
            self.opponent_difficulty = 0.0
        elif count < 7000:
            self.training_phase = 0  # Phase avancée (3000 épisodes)
            self.opponent_difficulty = 0.0
        elif count < 10000:
            self.training_phase = 0  # Phase expert (3000 épisodes)
            self.opponent_difficulty = 0.0
        else:
            self.training_phase = 0  # Phase maître (infini)
            self.opponent_difficulty = 0.0
        
    def reset(self, seed=None, options=None):
        """Réinitialise l'environnement pour un nouvel épisode."""
        super().reset(seed=seed)
        
        # Créer les objets du jeu
        self.table = Table()
        self.net = Net()
        net_center = WIDTH // 2

        print(self.training_phase, self.episode_count)
        if self.game_mode:
            # Mode jeu: positions fixes, pas de curriculum
            self._speed_factor = 1.0
            if self.agent_side == "left":
                self.agent_paddle = Paddle(150, TABLE_Y - 150, x_min=0, x_max=net_center)
                self.opponent_paddle = Paddle(TABLE_Y + TABLE_WIDTH_PX//2 + OUT_MARGIN + RACKET_HEIGHT_PX//2, TABLE_Y - 150, x_min=net_center, x_max=WIDTH)
            else:
                self.agent_paddle = Paddle(TABLE_Y + TABLE_WIDTH_PX//2 + OUT_MARGIN + RACKET_HEIGHT_PX//2, TABLE_Y - 150, x_min=net_center, x_max=WIDTH)
                self.opponent_paddle = Paddle(150, TABLE_Y - 150, x_min=0, x_max=net_center)
        else:
            # === Curriculum: Ajuster la distance et vitesse initiale selon la phase ===
            phase_configs = {
                0: {'speed_factor': 1},   # Balle ULTRA lente pour vraiment apprendre les bases
                1: {'speed_factor': 1},   # Balle lente
                2: {'speed_factor': 1},   # Balle modérée
                3: {'speed_factor': 1},  # Balle presque normale
                4: {'speed_factor': 1.0},   # Vitesse normale (phase maître)
            }
            config = phase_configs.get(self.training_phase, phase_configs[4])
            self._speed_factor = config['speed_factor']
            # Créer les raquettes selon le côté de l'agent
            # Distance progressive: Phase 0=très loin (forcer apprentissage), Phase 4=normal
            # Augmenté pour forcer l'IA à apprendre l'approche APRÈS le rebond, pas avant
            distance_offset = min(self.training_phase, 4) * 150  # Max 600px de distance (augmenté de 120 -> 150)
            if self.agent_side == "left":
                px, py = net_center - 300 - random.randint(0, 250), TABLE_Y - random.randint(30, 500)
                paddle_x = net_center - 300 - distance_offset  # Augmenté de 240 -> 300
                self.agent_paddle = Paddle(px, py, x_min=0, x_max=net_center)
                self.opponent_paddle = Paddle(TABLE_Y + TABLE_WIDTH_PX//2 + OUT_MARGIN + RACKET_HEIGHT_PX//2, 10, x_min=net_center, x_max=WIDTH)
            else:
                paddle_x = net_center + 300 + distance_offset  # Augmenté de 240 -> 300
                self.agent_paddle = Paddle(paddle_x, HEIGHT // 2 - 30, x_min=net_center, x_max=WIDTH)
                self.opponent_paddle = Paddle(50, 10, x_min=0, x_max=net_center)
        
        # Randomiser le service
        self.is_agent_service = True
        
        if self.is_agent_service:
            # L'agent sert
            if self.agent_side == "left":
                if self.game_mode == False:
                    prob = random.randint(1, 10) # de 1 à 10 (inclus)
                    if (prob == 1) or (prob == 2):
                        self.ball = spawn_ball_left(self.table, game_mode=self.game_mode, train_phase=self.training_phase)
                    else: 
                        self.ball = spawn_ball_right(self.table, game_mode=self.game_mode, train_phase=self.training_phase)
                        self.ball.service = None
                        self.ball.last_hit_by = "right"
                else:
                    self.ball = spawn_ball_left(self.table, game_mode=self.game_mode, train_phase=self.training_phase)
            else:
                self.ball = spawn_ball_right(self.table, game_mode=self.game_mode, train_phase=self.training_phase)
        else:
            # L'adversaire sert
            if self.agent_side == "left":
                self.ball = spawn_ball_right(self.table, game_mode=self.game_mode, train_phase=self.training_phase)
            else:
                self.ball = spawn_ball_left(self.table, game_mode=self.game_mode, train_phase=self.training_phase)

        # Mode spawn statique: pas de gravité tant qu'aucun hit
        if self.static_spawn:
            self.ball.gravity_enabled = False
            self.ball.vel[:] = 0
            self.ball.angular_speed = 0

        # Respect strict des règles : pas de rebond forcé en phase 0
        # Le serveur doit attendre le rebond avant de frapper (même au service)
        
        self.ball_in_play = True
        self.steps = 0
        self.last_hit_by = None
        self.agent_hits = 0  # Compteur de frappes de l'agent
        
        # Reset de l'accumulation des rewards pour ce nouvel épisode
        self.episode_reward_accumulation = 0.0

        # Reset de la proximité du passage de balle à TABLE_Y
        self.table_cross_proximity = None
        
        # Reset du tracking de distance pour reward shaping
        self.prev_dist_to_ball = None
        
        # Reset des flags de récompenses
        self.bounce_reward_given = False # flag temporaire qui s'active et se désactive la balle touche la table adverse, pour donner une récompense une seule fois
        self.proximity_reward_given = False
        self.fault_volley = False # touche la balle en volée
        self.opponent_fault_volley = False # l'adversaire touche la balle en volée
        self.double_hit_fault = False
        self.service_fault = False
        self.pending_hit_reward = False
        self.pending_wrong_direction = False  # Frappe dans la mauvaise direction (pending)
        self.ball_out_result = None # flag temporaire qui s'active et se désactive quand on touche une balle, pour donner une récompense une seule fois
        self.point_winner_side = None
        self.coef_speed = 0.0
        
        observation = self._get_observation()
        info = {}
        

        return observation, info
    
    def step(self, action, opponent_action=None):
        """
        Exécute une action et retourne le nouvel état (API Gymnasium/SB3).
        
        Args:
            action: Action de l'agent principal
            opponent_action: (Optionnel) Action de l'adversaire. 
                             Si None, utilise l'IA interne _get_opponent_action().
        
        Returns:
            observation, reward, terminated, truncated, info
        """

        # affichage vel joueur droite
        #print(f"Vel joueur droite dans step(): {self.opponent_paddle.vel}")
        
        self.steps += 1
        self.point_winner_side = None  # reset du vainqueur pour ce step
        agent_is_left = (self.agent_side == "left")
        agent_paddle_side = 'left' if agent_is_left else 'right'

        self.coef_speed = 0.0
        
        # === Appliquer l'action de l'agent ===
        self._apply_action(self.agent_paddle, action)
        
        # === Appliquer l'action de l'adversaire ===
        if opponent_action is None:
            # IA simple par défaut
            actual_opponent_action = self._get_opponent_action()
        else:
            # Action fournie (pour le self-play ou 2ème agent)
            actual_opponent_action = opponent_action
            
        self._apply_action(self.opponent_paddle, actual_opponent_action)
        

        # === Mettre à jour la physique ===
        dt = 1.0 / FPS
        self.agent_paddle.update(dt, speed_factor=self._speed_factor)
        self.opponent_paddle.update(dt, speed_factor=self._speed_factor)
        
        if self.ball_in_play:
            # Sub-stepping pour éviter le tunneling (balle qui traverse la raquette)
            n_substeps = 4
            dt_sub = dt / n_substeps
            
            for _ in range(n_substeps):
                self.ball.update(dt=dt_sub, speed_factor=self._speed_factor)
                
                # === DETECTION BALLE OUT ===
                # Deux cas distincts :
                # 1. Pas de rebond adverse : balle sort des limites + marge -> point fini
                # 2. Avec rebond adverse : balle sort des limites de l'écran -> point fini
                if self.ball_out_result is None:
                    hitter = self.ball.last_hit_by
                    
                    # Cas où on ne sait pas qui a frappé (au début)
                    if hitter not in ('left', 'right'):
                        hitter = None
                    
                    # Vérifier s'il y a eu un rebond sur le côté adverse
                    has_valid_bounce = False
                    if self.ball.bounces_right > 0 and hitter == 'left':
                        has_valid_bounce = True
                    elif self.ball.bounces_left > 0 and hitter == 'right':
                        has_valid_bounce = True
                    
                    # === CAS 1 : PAS DE REBOND ADVERSE (détection précoce avec marges) ===
                    # vérifier si la balle est de l'autre côté que du frappeur et qu'il n'y a pas de rebond
                    if self.ball_side is not None and hitter is not None and self.ball_side != hitter and not has_valid_bounce:
                    
                        table_left_limit = self.table.x - OUT_MARGIN
                        table_right_limit = self.table.x + self.table.width + OUT_MARGIN
                        
                        side_out = None
                        if self.ball.pos[0] < table_left_limit:
                            side_out = 'left'
                        elif self.ball.pos[0] > table_right_limit:
                            side_out = 'right'
                        
                        if side_out is not None:
                            # Pas de rebond adverse -> faute du frappeur
                            winner_side = 'right' if hitter == 'left' else 'left'
                            self.point_winner_side = winner_side
                            
                            # Traduire en résultat pour l'agent
                            agent_wins = (
                                (winner_side == 'left' and self.agent_side == 'left') or
                                (winner_side == 'right' and self.agent_side == 'right')
                            )
                            self.ball_out_result = 'win' if agent_wins else 'loss'
                    
                    # === CAS 2 : AVEC REBOND ADVERSE (attendre sortie complète de l'écran) ===
                    elif has_valid_bounce and hitter is not None:
                        # Limites de l'écran (sans marge)
                        screen_left = 0
                        screen_right = WIDTH
                        
                        side_out = None
                        if self.ball.pos[0] < screen_left:
                            side_out = 'left'
                        elif self.ball.pos[0] > screen_right:
                            side_out = 'right'
                        
                        if side_out is not None:
                            # Il y a eu un rebond côté adverse
                            if side_out == hitter:
                                # Sortie du côté du frappeur -> faute du frappeur
                                winner_side = 'right' if hitter == 'left' else 'left'
                            else:
                                # Sortie du côté du receveur -> point au frappeur
                                winner_side = hitter

                            self.point_winner_side = winner_side
                            
                            # Traduire en résultat pour l'agent
                            agent_wins = (
                                (winner_side == 'left' and self.agent_side == 'left') or
                                (winner_side == 'right' and self.agent_side == 'right')
                            )
                            self.ball_out_result = 'win' if agent_wins else 'loss'

                # Sortie par le bas (sous la table) détectée dans step
                if self.ball_out_result is None and self.ball.pos[1] > HEIGHT:
                    last = self.ball.last_hit_by
                    
                    # Déterminer de quel côté la balle tombe (avec offset adaptatif)
                    # Si la balle va à droite: offset +ADAPTIVE_BOUNDARY_OFFSET, si elle va à gauche: offset -ADAPTIVE_BOUNDARY_OFFSET
                    velocity_offset = ADAPTIVE_BOUNDARY_OFFSET if self.ball.vel[0] > 0 else -ADAPTIVE_BOUNDARY_OFFSET
                    net_center = WIDTH // 2 + velocity_offset
                    ball_falls_on_left = self.ball.pos[0] < net_center
                    
                    # Déterminer qui gagne selon où tombe la balle et qui a frappé en dernier
                    if last == 'left':
                        # Le joueur gauche a frappé en dernier
                        if ball_falls_on_left:
                            # Tombe de son côté → il perd
                            winner_side = 'right'
                        else:
                            # Tombe du côté adverse → il gagne
                            winner_side = 'left'
                    elif last == 'right':
                        # Le joueur droite a frappé en dernier
                        if ball_falls_on_left:
                            # Tombe du côté adverse → il gagne
                            winner_side = 'right'
                        else:
                            # Tombe de son côté → il perd
                            winner_side = 'left'
                    
                    self.point_winner_side = winner_side
                    
                    # Traduire en résultat pour l'agent
                    agent_wins = (
                        (winner_side == 'left' and agent_is_left) or
                        (winner_side == 'right' and not agent_is_left)
                    )
                    self.ball_out_result = 'win' if agent_wins else 'loss'
                
                # Si le point est terminé par un OUT, on arrête la physique/collisions
                if self.ball_out_result is not None:
                    continue

                # Collisions (table/net) avant de vérifier le changement de côté, 
                # pour que les rebonds soient comptés avant le contrôle de service.
                check_table_collision(self.ball, self.table)
                check_ball_net(self.ball, self.net)

                # Détecter le changement de côté de la balle (avec offset adaptatif basé sur la vélocité)
                # Si la balle va à droite: offset +ADAPTIVE_BOUNDARY_OFFSET, si elle va à gauche: offset -ADAPTIVE_BOUNDARY_OFFSET
                velocity_offset = ADAPTIVE_BOUNDARY_OFFSET if (self.ball_side == "left") else -ADAPTIVE_BOUNDARY_OFFSET
                net_center = WIDTH // 2 + velocity_offset
                if self.ball is not None:
                    current_side = 'left' if self.ball.pos[0] < net_center else 'right'
                else:
                    current_side = 'left'  # par défaut
                
                # Si la balle change de côté
                if self.ball_side is not None and current_side != self.ball_side:

                    
                    server_side = self.ball.service
                    if server_side == 'left' and current_side == 'right':
                        if self.ball.bounces_left != 1:
                            self.service_fault = True
                    elif server_side == 'right' and current_side == 'left':
                        if self.ball.bounces_right != 1:
                            self.service_fault = True
                    
                    # Fin du service : si la balle passe de l'autre côté, on désactive le flag service
                    if hasattr(self.ball, 'last_hit_by'):
                        self.ball.previous_hit_by = self.ball.last_hit_by
                        if self.ball.service is not None and self.ball.previous_hit_by != self.ball.service:
                            self.ball.service = None

                self.ball_side = current_side
                
                # === COLLISIONS AVEC RAQUETTES ===
                
                # Collision avec raquette agent
                if self.agent_paddle.can_hit:
                    ball_hit_agent = self._check_paddle_collision(self.agent_paddle, "agent")
                    if ball_hit_agent:
                        self.bounce_reward_given = False
                        self.proximity_reward_given = False
                        agent_paddle_side = 'left' if self.agent_side == 'left' else 'right'
                        
                        # Sauvegarder qui a frappé avant (pour détecter volley)
                        if hasattr(self.ball, 'last_hit_by'):
                            self.ball.previous_hit_by = self.ball.last_hit_by
                        
                        # DÉTECTION VOLLEY : Frapper sans rebond sur notre côté (même au service)
                        our_bounces = self.ball.bounces_left if agent_paddle_side == 'left' else self.ball.bounces_right
                        if our_bounces == 0:
                            # FAUTE : frapper sans rebond sur notre table
                            self.fault_volley = True
                        
                        # DÉTECTION DIRECTION INCORRECTE : Frapper la balle vers son propre camp
                        # Si agent à gauche : balle doit aller vers la droite (vel[0] > 0)
                        # Si agent à droite : balle doit aller vers la gauche (vel[0] < 0)
                        ball_vel_x = self.ball.vel[0]
                        if agent_paddle_side == 'left' and ball_vel_x < -50:  # Seuil pour éviter faux positifs sur petits angles
                            self.pending_wrong_direction = True
                        elif agent_paddle_side == 'right' and ball_vel_x > 50:
                            self.pending_wrong_direction = True

                        max_vel = 1000
                        if not self.pending_wrong_direction:
                            ball_speed = np.sqrt(self.ball.vel[0]**2 + self.ball.vel[1]**2)
                            if ball_speed >= max_vel:
                                coef_speed = 1
                            else:
                                coef_speed = ball_speed / max_vel
                        
                        # Réactiver la gravité au premier contact si spawn statique
                        self.ball.gravity_enabled = True
                        
                        self.pending_hit_reward = True
                        self.agent_hits += 1
                        self.ball.last_hit_by = agent_paddle_side
                        
                        # Reset can_hit de l'adversaire et les rebonds de l'autre côté
                        self.opponent_paddle.can_hit = True
                        opponent_side = 'right' if agent_paddle_side == 'left' else 'left'
                        if opponent_side == 'left':
                            self.ball.bounces_left = 0
                        else:
                            self.ball.bounces_right = 0
                
                # Collision avec raquette adversaire
                if self.opponent_paddle.can_hit:
                    ball_hit_opponent = self._check_paddle_collision(self.opponent_paddle, "opponent")
                    if ball_hit_opponent:

                        opponent_paddle_side = 'right' if self.agent_side == 'left' else 'left'
                        
                        # Sauvegarder qui a frappé avant
                        if hasattr(self.ball, 'last_hit_by'):
                            self.ball.previous_hit_by = self.ball.last_hit_by

                        # DÉTECTION VOLLEY ADVERSAIRE : frapper sans rebond sur son côté (même au service)
                        opponent_bounces = self.ball.bounces_right if opponent_paddle_side == 'right' else self.ball.bounces_left
                        if opponent_bounces == 0:
                            self.opponent_fault_volley = True
                        
                        # Réactiver la gravité au premier contact si spawn statique
                        self.ball.gravity_enabled = True
                        
                        self.ball.last_hit_by = opponent_paddle_side
                        
                        # Reset can_hit de l'agent et les rebonds de l'autre côté
                        self.agent_paddle.can_hit = True
                        agent_paddle_side = 'left' if self.agent_side == 'left' else 'right'
                        if agent_paddle_side == 'left':
                            self.ball.bounces_left = 0
                        else:
                            self.ball.bounces_right = 0
        
        # Détecter le passage à TABLE_Y (descente) pour un reward de proximité
        if self.proximity_reward_given == False:
            if self.ball.pos[1] - 5 < TABLE_Y - 10 <= self.ball.pos[1] + 5:
                if self.ball.last_hit_by == agent_paddle_side:
                    #print(f"--------------------------------------------------------- y == table_y, {self.ball.last_hit_by =}, {agent_paddle_side =     }")
                    max_distance = WIDTH / 2
                    if self.ball.pos[0] < (WIDTH / 2):
                        #print("gauche")
                        self.table_cross_proximity = (max_distance - abs(self.ball.pos[0] - WIDTH / 2)) / max_distance
                    else:
                        #print("droite")
                        self.table_cross_proximity = (max_distance - abs(self.ball.pos[0] - (TABLE_Y + TABLE_WIDTH_PX))) / max_distance

        # === DÉTECTION DES FAUTES POUR LES DEUX CÔTÉS ===
        terminated = False
        truncated = False
        faults = {
            'volley_left': False,
            'volley_right': False,
            'double_bounce_left': False,
            'double_bounce_right': False,
            'out': False,
            'service_fault': False,
            'ball_out_bottom': False,
            'timeout': False,
        }
        
        # Volley (obstruction) - frapper sans rebond sur notre table
        if self.fault_volley or self.opponent_fault_volley:
            agent_is_left = (self.agent_side == "left")
            if self.fault_volley:
                # L'agent a frappé sans rebond -> il perd
                faults['volley_left' if agent_is_left else 'volley_right'] = True
                self.point_winner_side = 'right' if agent_is_left else 'left'
            else:
                # L'adversaire a frappé sans rebond -> il perd
                faults['volley_right' if agent_is_left else 'volley_left'] = True
                self.point_winner_side = 'left' if agent_is_left else 'right'
            terminated = True
            self.ball_in_play = False
            self._update_scores("Volley/Obstruction - pas de rebond avant frappe!")
        
        # Double rebond - vérifier pour les deux côtés
        if not terminated and self.ball.bounces_left >= 2:
            faults['double_bounce_left'] = True
            self.point_winner_side = 'right'
            terminated = True
            self.ball_in_play = False
            self._update_scores("Double rebond gauche!")
        
        if not terminated and self.ball.bounces_right >= 2:
            faults['double_bounce_right'] = True
            self.point_winner_side = 'left'
            terminated = True
            self.ball_in_play = False
            self._update_scores("Double rebond droite!")
        
        # Balle sortie (latérale)
        if not terminated and self.ball_out_result is not None:
            faults['out'] = True
            # point_winner_side déjà défini dans la détection OUT
            terminated = True
            self.ball_in_play = False
            self._update_scores("Out!")
        
        # Faute de service
        if not terminated and self.service_fault:
            faults['service_fault'] = True
            # point_winner_side à déterminer selon le serveur
            server_side = self.ball.service
            if server_side in ('left', 'right'):
                self.point_winner_side = 'right' if server_side == 'left' else 'left'
            terminated = True
            self.ball_in_play = False
            self._update_scores("Service invalide!")
        
        observation = self._get_observation()
        info = {
            "steps": self.steps,
            "agent_hits": self.agent_hits,
            "winner_side": self.point_winner_side,  # 'left', 'right' ou None
            "faults": faults,
            "score_left": self.score_left,
            "score_right": self.score_right,
            "point_message": self.last_point_message,
            "ball_on_agent_side": self._is_ball_on_agent_side(),
            "ball_bounces_agent": self._get_agent_side_bounces(),
            "ball_bounces_opponent": self._get_opponent_side_bounces(),
            "wrong_direction": self.pending_wrong_direction,  # Frappe dans la mauvaise direction (pending)
            "table_cross_proximity": self.table_cross_proximity,
        }

        # Calcul du reward (shaping + terminal) intégré à la step
        reward = self._compute_shaped_reward(info)

        #print("Reward step:", reward)
        
        # Rendu si demandé
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, paddle, action):
        """
        Applique une action continue directement à la vélocité de la raquette.
        
        Système continu (pas de discrétisation):
        - action_x ∈ [-1, 1] → paddle.vel[0] = action_x * paddle.speed
        - action_y ∈ [-1, 1] → paddle.vel[1] = action_y * paddle.speed
        - action_rot ∈ [-1, 1] → paddle.angle += action_rot * max_rotation_speed * dt
        
        Si player1_mouse_control ou player2_mouse_control est True pour cette raquette,
        on n'applique que la rotation (mouvement géré par la souris).
        """
        move_x, move_y, rotate = action
        
        # Déterminer si cette raquette est contrôlée par la souris
        is_mouse_controlled = False
        if paddle == self.agent_paddle and self.player1_mouse_control:
            is_mouse_controlled = True
        elif paddle == self.opponent_paddle and self.player2_mouse_control:
            is_mouse_controlled = True
        
        # Si contrôlé par souris, ne pas écraser la vélocité
        if not is_mouse_controlled:
            # Contrôle CONTINU: appliquer directement l'action à la vélocité
            # action ∈ [-1, 1] * speed → paddle bouge proportionnellement à l'action
            paddle.vel[0] = float(move_x) * paddle.speed
            paddle.vel[1] = float(move_y) * paddle.speed
        
        # Rotation CONTINUE (dt = magnitude de l'action normalisée)
        # action_rot ∈ [-1, 1] → angle change proportionnellement
        # max_rotation_speed ≈ 360°/s avec action_rot=1
        max_rotation_speed = 360.0  # degrés par seconde
        dt = 1.0 / FPS
        
        if rotate != 0:
            # angle += action_rot * max_rotation_speed * dt
            paddle.angle += float(rotate) * max_rotation_speed * dt
            paddle.angle %= 360
    
    def _get_opponent_action(self):
        """
        IA simple pour l'adversaire : suit la balle en Y.
        À remplacer par un autre agent entraîné pour le self-play.
        """
        if not self.ball_in_play:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Suivre la balle en Y
        paddle_center_y = self.opponent_paddle.pos[1] + self.opponent_paddle.height / 2
        ball_y = self.ball.pos[1]
        
        move_y = 0.0
        if ball_y < paddle_center_y - 20:
            move_y = -1.0  # Monter
        elif ball_y > paddle_center_y + 20:
            move_y = 1.0   # Descendre
        
        move_x = 0.0
        
        #return np.array([move_x, move_y, 0.0], dtype=np.float32) # adversaire qui joue pas
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    def _check_paddle_collision(self, paddle, who):
        """Vérifie la collision balle-raquette et met à jour last_hit_by."""

        old_pos = self.ball.pos.copy()
        check_ball_paddle(self.ball, paddle, None)
        
        # Si la position a changé, c'est qu'il y a eu collision
        if not np.array_equal(old_pos, self.ball.pos):
            self.last_hit_by = who
            return True
        return False
    
    def _compute_shaped_reward(self, info):
        """
        Compute normalized reward in range [-1, 1].
        Rewards are scaled by dividing by 15.0 (the max absolute reward value).
        """
        reward = 0.0
        REWARD_SCALE = 10.0  # Normalisation : tous les rewards divisés par cette valeur
        
        agent_is_left = (self.agent_side == "left")
        winner_side = info.get('winner_side')
        faults = info.get('faults', {})
        
        # === 1. RÉCOMPENSES TERMINALES (LA PRIORITÉ ABSOLUE) ===
        if winner_side is not None:
            agent_wins = (
                (winner_side == 'left' and agent_is_left) or
                (winner_side == 'right' and not agent_is_left)
            )
            
            if agent_wins:
                # Doit être > (Hit + Bounce + Shaping accumulé)
                reward = 10.0 / REWARD_SCALE  # = 1.0
                log_msg = "🟢 WIN"
            else:
                # La défaite doit faire mal pour motiver la défense (symétrique avec victoire)
                reward = -10.0 / REWARD_SCALE  # = -1.0 (équilibré avec victoire)
                if faults.get('double_bounce_left', False):
                    reward -= 1.0 / REWARD_SCALE  # -0.1 extra
                log_msg = "🔴 LOSS"
            
            # Accumuler le reward
            self.episode_reward_accumulation += reward
            
            if self.render_mode == "human":
                print(f"    {log_msg} | Reward: {reward:.3f} | Episode Sum: {self.episode_reward_accumulation:.3f}")
            
            return reward # On coupe ici, pas de shaping sur la frame finale
        
        # === 2. RÉCOMPENSES INTERMÉDIAIRES (SHAPING) ===
        if self.ball_in_play:
            ball_x, ball_y = self.ball.pos
            paddle_center_x = self.agent_paddle.pos[0] + self.agent_paddle.width / 2
            paddle_center_y = self.agent_paddle.pos[1] + self.agent_paddle.height / 2
            
            dist_to_ball = np.sqrt((ball_x - paddle_center_x)**2 + (ball_y - paddle_center_y)**2)
            ball_on_agent_side = info.get('ball_on_agent_side', False)

            # Rebond côté agent (condition pour autoriser l'approche)
            our_bounces = self.ball.bounces_left if agent_is_left else self.ball.bounces_right

            # A. AVANT REBOND : Rester neutre (pas de guidage, pas de pénalité)
            # L'agent doit se positionner librement sans contrainte
            if ball_on_agent_side and our_bounces == 0:
                # Reset du tracking : on ne guide pas, mais on ne pénalise pas non plus
                self.prev_dist_to_ball = None
            
            # B. APRÈS REBOND : Guidage actif vers la balle
            elif ball_on_agent_side and our_bounces > 0:
                # Bonus d'approche (Delta seulement, pas de pénalité de distance absolue)
                if self.prev_dist_to_ball is not None:
                    improvement = self.prev_dist_to_ball - dist_to_ball
                    # On récompense uniquement si on se rapproche
                    if improvement > 0:
                        reward += (improvement * 0.005) / REWARD_SCALE  # Augmenté de 0.003 -> 0.005
                self.prev_dist_to_ball = dist_to_ball
            else:
                # Tant que la balle n'a pas rebondi chez nous, on ne guide pas l'approche
                self.prev_dist_to_ball = None

            # B. Bonus d'Alignement (Vitesse directionnelle) - APRÈS rebond
            paddle_vel_x, paddle_vel_y = self.agent_paddle.vel
            paddle_speed = np.sqrt(paddle_vel_x**2 + paddle_vel_y**2)
            
            if paddle_speed > 10 and ball_on_agent_side and our_bounces > 0:  # Seuil augmenté de 5 -> 10
                dir_to_ball_x = (ball_x - paddle_center_x) / (dist_to_ball + 1e-6)
                dir_to_ball_y = (ball_y - paddle_center_y) / (dist_to_ball + 1e-6)
                
                vel_norm_x = paddle_vel_x / (paddle_speed + 1e-6)
                vel_norm_y = paddle_vel_y / (paddle_speed + 1e-6)
                
                alignment = vel_norm_x * dir_to_ball_x + vel_norm_y * dir_to_ball_y
                
                if alignment > 0.5:  # Seuil modéré pour permettre l'apprentissage précoce
                    reward += (0.01 * alignment) / REWARD_SCALE  # Réduit de 0.02 -> 0.01

            # === 3. ÉVÉNEMENTS CLÉS (Jalons) ===
            # Ces récompenses doivent être significatives mais inférieures à la Victoire
            
            # BONUS REBOND : Frapper APRÈS exactement 1 rebond (comportement correct)
            if self.pending_hit_reward:
                our_bounces_at_hit = self.ball.bounces_left if agent_is_left else self.ball.bounces_right
                
                if our_bounces_at_hit == 1:
                    # EXCELLENT : frappe après 1 rebond
                    reward += 2.0 / REWARD_SCALE  # = 0.2 (5 * moins que la victoire)
                    self.pending_hit_reward = False
            
            # PÉNALITÉ DIRECTION INCORRECTE : Frapper vers son propre camp (comportement très mauvais)
            if self.pending_wrong_direction:
                reward -= 8.0 / REWARD_SCALE  # = -0.8 (pénalité FORTE - presque aussi grave qu'une défaite)
                self.pending_wrong_direction = False  # Consommer le flag (une seule fois par frappe)
                if self.render_mode == "human":
                    print("    ⚠️ WRONG DIRECTION HIT! Ball sent backwards!")
            else:
                reward += (3.0 * self.coef_speed) / REWARD_SCALE # favorise une balle rapide
            

            
            # Mettre la balle chez l'adversaire (rebond valide) : C'est très bien
            ball_bounces_opponent = info.get('ball_bounces_opponent', 0)
            agent_hits = info.get('agent_hits', 0)
            table_cross_proximity = info.get('table_cross_proximity')
            
            if agent_hits > 0 and ball_bounces_opponent > 0:
                if not self.bounce_reward_given:
                    reward += 8.0 / REWARD_SCALE  # = 0.4 (envoie un signal fort)
                    self.bounce_reward_given = True
            elif agent_hits > 0 and table_cross_proximity is not None:
                if not self.proximity_reward_given:
                    # Proximité du rebond adverse même si pas de rebond valide
                    #print('='*70)
                    #print(f"Table cross proximity: {table_cross_proximity}")
                    reward += (5.0 * table_cross_proximity) / REWARD_SCALE  # 0..0.5
                    self.proximity_reward_given = True
        
        # Accumuler les rewards intermédiaires
        self.episode_reward_accumulation += reward
        
        #print(f"Reward step: {reward:.3f}")
        return reward
    """
        # === RÉCOMPENSES INTERMÉDIAIRES (point en cours) ===
        if winner_side is None and self.ball_in_play:
            ball_x = self.ball.pos[0]
            ball_y = self.ball.pos[1]
            
            # Position de la raquette agent
            paddle_center_x = self.agent_paddle.pos[0] + self.agent_paddle.width / 2
            paddle_center_y = self.agent_paddle.pos[1] + self.agent_paddle.height / 2
            
            # Vélocité de la raquette
            paddle_vel_x = self.agent_paddle.vel[0]
            paddle_vel_y = self.agent_paddle.vel[1]
            paddle_speed = np.sqrt(paddle_vel_x**2 + paddle_vel_y**2)
            
            # Direction vers la balle
            dir_to_ball_x = ball_x - paddle_center_x
            dir_to_ball_y = ball_y - paddle_center_y
            dist_to_ball = np.sqrt(dir_to_ball_x**2 + dir_to_ball_y**2)

            
            ball_is_playable = ball_y > (TABLE_Y - 50)
            
            ball_on_agent_side = info.get('ball_on_agent_side', False)

            # === Proximité avec la balle ===
            if ball_on_agent_side and ball_is_playable:
                # Pénalité légère sur la distance absolue (guide général)
                reward -= dist_to_ball * 0.001
                
                # Bonus sur l'amélioration de la distance (delta), normalisé par la largeur
                # Encourage X/Y à se diriger vers la balle à chaque step
                if self.prev_dist_to_ball is not None:
                    improvement = max(0.0, self.prev_dist_to_ball - dist_to_ball)
                    reward += (improvement / WIDTH) * 1.0  # coef petit (~<=0.02 par step typique)
                self.prev_dist_to_ball = dist_to_ball

            # --- 1. SIGNAL DIRECTIONNEL (faible, guide move_x/move_y) ---
            if paddle_speed > 10 and dist_to_ball > 1.0:
                # Normaliser les vecteurs
                dir_to_ball_norm_x = dir_to_ball_x / max(dist_to_ball, 1e-6)
                dir_to_ball_norm_y = dir_to_ball_y / max(dist_to_ball, 1e-6)
                
                paddle_vel_norm_x = paddle_vel_x / max(paddle_speed, 1e-6)
                paddle_vel_norm_y = paddle_vel_y / max(paddle_speed, 1e-6)
                
                # Produit scalaire : alignement entre direction et vélocité
                alignment = paddle_vel_norm_x * dir_to_ball_norm_x + paddle_vel_norm_y * dir_to_ball_norm_y
                
                # alignment ∈ [-1, 1]
                # +1 = va directement vers la balle, -1 = à l'opposé
                if alignment > 0:
                    reward += 0.01 * alignment
                else:
                    reward += 0.005 * alignment  # pénalité plus faible si à l'opposé
            
            # --- 2. TOUCHE DE BALLE (1er jalon important) ---
            if self.pending_hit_reward:
                reward += 3.0  # Récompense claire pour le touch
                self.pending_hit_reward = False
            
            # --- 3. REBOND ADVERSAIRE (2e jalon - bien frappé) ---
            ball_bounces_opponent = info.get('ball_bounces_opponent', 0)
            agent_hits = info.get('agent_hits', 0)
            if agent_hits > 0 and ball_bounces_opponent > 0:
                if not self.bounce_reward_given:
                    reward += 5.0  # Preuve d'un bon coup
                    self.bounce_reward_given = True
        
        return reward
    """
    
    def _get_distance_ball_paddle_normalized(self, paddle):
        """Calcule la distance normalisée entre la balle et une raquette."""
        if not self.ball_in_play or self.ball is None:
            return 1.0  # Distance maximale si la balle n'est pas en jeu
        
        paddle_center_x = paddle.pos[0] + paddle.width / 2
        paddle_center_y = paddle.pos[1] + paddle.height / 2
        ball_x = self.ball.pos[0]
        ball_y = self.ball.pos[1]
        
        distance = np.sqrt((paddle_center_x - ball_x)**2 + (paddle_center_y - ball_y)**2)
        
        # Normaliser la distance (max ~WIDTH)
        normalized_distance = np.clip(distance / WIDTH, 0.0, 1.0)
        
        return normalized_distance
    
    def _position_to_grid_index(self, x, y, grid_size=16):
        """Convertit une position (x, y) en index de cellule de grille (16×16 = 256 cellules)."""
        # Normaliser les coordonnées dans [0, 1]
        x_norm = np.clip(x / WIDTH, 0.0, 0.999)
        y_norm = np.clip(y / HEIGHT, 0.0, 0.999)
        
        # Calculer l'index de cellule
        col = int(x_norm * grid_size)
        row = int(y_norm * grid_size)
        
        return row * grid_size + col

    def _get_observation(self):
        """
        Retourne l'observation avec spatial embeddings.
        
        Format: dict avec
            - 'ball_idx': index de cellule grille pour position balle (int 0-255, grille 16×16)
            - 'paddle_idx': index de cellule grille pour position raquette agent (int 0-255)
            - 'angle_idx': index d'angle discret pour raquette agent (int 0-15, cyclique)
            - 'continuous': array de features continues [18 valeurs]:
                [0-1]   Vitesse balle (vx, vy) normalisée
                [2]     Spin balle normalisé
                [3-4]   Vitesse raquette agent (vx, vy) normalisée
                [5]     Balle de notre côté ? (1 = oui, -1 = non)
                [6]     Balle vient vers nous ? (1 = oui, -1 = non)
                [7]     Rebonds sur notre côté (0, 0.5, 1)
                [8]     Rebonds côté adverse (0, 0.5, 1)
                [9]     Est-ce un service ? (1 = oui, -1 = non)
                [10-11] Distances normalisées balle→raquette (dx, dy) dans [-1, 1]
                [12-13] Position raquette agent continue (x, y) normalisée
                [14-15] Vitesse relative balle-raquette (vx_rel, vy_rel) pour timing
                [16-17] sin(angle) et cos(angle) pour continuité circulaire
                [18] Ai-je le droit de frapper ? (1=oui, -1=non)
        """
        agent_is_left = (self.agent_side == "left")
        continuous = np.zeros(19, dtype=np.float32)
        
        # Centre de la raquette agent (utilisé pour distances relatives)
        paddle_center_x = self.agent_paddle.pos[0] + self.agent_paddle.width / 2
        paddle_center_y = self.agent_paddle.pos[1] + self.agent_paddle.height / 2
        
        if self.ball_in_play and self.ball is not None:
            ball_x = self.ball.pos[0]
            ball_y = self.ball.pos[1]
            
            # Index de grille pour la balle
            ball_idx = self._position_to_grid_index(ball_x, ball_y)
            
            # Vitesse balle normalisée
            max_vel = 1000.0
            continuous[0] = np.clip(self.ball.vel[0] / max_vel, -1, 1)
            continuous[1] = np.clip(self.ball.vel[1] / max_vel, -1, 1)
            
            # Spin normalisé
            max_spin = 500.0
            continuous[2] = np.clip(self.ball.angular_speed / max_spin, -1, 1)
            
            # Balle de notre côté ?
            velocity_offset = ADAPTIVE_BOUNDARY_OFFSET if self.ball.vel[0] > 0 else -ADAPTIVE_BOUNDARY_OFFSET
            net_center_offset = WIDTH // 2 + velocity_offset
            ball_on_agent_side = (ball_x < net_center_offset) if agent_is_left else (ball_x >= net_center_offset)
            continuous[5] = 1.0 if ball_on_agent_side else -1.0
            
            # Balle vient vers nous ?
            ball_coming = (self.ball.vel[0] <= 0) if agent_is_left else (self.ball.vel[0] >= 0)
            continuous[6] = 1.0 if ball_coming else -1.0
            
            # Rebonds sur notre côté
            our_bounces = self.ball.bounces_left if agent_is_left else self.ball.bounces_right
            continuous[7] = min(our_bounces * 0.5, 1.0)
            
            # Rebonds côté adverse
            their_bounces = self.ball.bounces_right if agent_is_left else self.ball.bounces_left
            continuous[8] = min(their_bounces * 0.5, 1.0)
            
            # Est-ce un service ?
            continuous[9] = 1.0 if self.ball.service is not None else -1.0
            
            # Distances relatives balle→raquette normalisées dans [-1, 1]
            # dx = (ball_x - paddle_center_x) / (WIDTH/2), dy = (ball_y - paddle_center_y) / (HEIGHT/2)
            continuous[10] = np.clip(((ball_x - paddle_center_x) / (WIDTH / 2.0)), -1.0, 1.0)
            continuous[11] = np.clip(((ball_y - paddle_center_y) / (HEIGHT / 2.0)), -1.0, 1.0)
        else:
            # Balle pas en jeu, position par défaut au centre
            ball_idx = self._position_to_grid_index(WIDTH // 2, HEIGHT // 2)

            # Valeurs par défaut si pas de balle
            continuous[10] = 0.0
            continuous[11] = 0.0
        
        # Index de grille pour la raquette agent
        paddle_idx = self._position_to_grid_index(paddle_center_x, paddle_center_y)
        
        # Vitesse raquette agent normalisée
        max_paddle_vel = 500.0
        continuous[3] = np.clip(self.agent_paddle.vel[0] / max_paddle_vel, -1, 1)
        continuous[4] = np.clip(self.agent_paddle.vel[1] / max_paddle_vel, -1, 1)
        
        # Position Raquette Continue [-1, 1]
        continuous[12] = (paddle_center_x / WIDTH) * 2 - 1
        continuous[13] = (paddle_center_y / HEIGHT) * 2 - 1
        
        # NOUVEAU : Vitesse relative balle-raquette (pour timing d'impact)
        if self.ball_in_play and self.ball is not None:
            max_vel = 1000.0
            vel_rel_x = self.ball.vel[0] - self.agent_paddle.vel[0]
            vel_rel_y = self.ball.vel[1] - self.agent_paddle.vel[1]
            continuous[14] = np.clip(vel_rel_x / max_vel, -1, 1)
            continuous[15] = np.clip(vel_rel_y / max_vel, -1, 1)
        else:
            continuous[14] = 0.0
            continuous[15] = 0.0
        
        # NOUVEAU : sin/cos de l'angle pour continuité circulaire (359° proche de 1°)
        angle_rad = np.deg2rad(self.agent_paddle.angle)
        continuous[16] = np.sin(angle_rad)
        continuous[17] = np.cos(angle_rad)
        
        # NOUVEAU CRITIQUE : Ai-je le droit de frapper ? (rebond détecté)
        if self.ball_in_play and self.ball is not None:
            our_bounces = self.ball.bounces_left if agent_is_left else self.ball.bounces_right
            continuous[18] = 1.0 if our_bounces > 0 else -1.0
        else:
            continuous[18] = -1.0  # Pas de balle = pas le droit de frapper

        # Angle raquette en index discret cyclique [0-15]
        # Angle [-180, 180] -> 16 bins (22.5 degrés par bin)
        # Normaliser: (-angle/180) -> [-1, 1], puis +1 -> [0, 2], *8 -> [0, 16]
        angle_normalized = (self.agent_paddle.angle % 360) / 22.5  # [0, 16)
        angle_idx = int(angle_normalized) % 16  # Assurer cyclicité

        #print(f"Obs - BallIdx: {ball_idx}, PaddleIdx: {paddle_idx}, AngleIdx: {angle_idx}, Continuous: {continuous}")
        
        # Retourner au format SB3 Dict (chaque valeur doit être un numpy array)
        return {
            'ball_idx': np.array([ball_idx], dtype=np.int64),
            'paddle_idx': np.array([paddle_idx], dtype=np.int64),
            'angle_idx': np.array([angle_idx], dtype=np.int64),
            'continuous': continuous
        }

    # ANCIENNE VERSION (2 features simples) - GARDÉE EN COMMENTAIRE
    # def _get_observation(self):
    #     Retourne l'observation normalisée (18 valeurs).
        
    #     Structure:
    #     [0-1]   Position balle (x, y)
    #     [2-3]   Vitesse balle (vx, vy)
    #     [4]     Spin balle
    #     [5-6]   Position raquette agent (x, y)
    #     [7-8]   Vitesse raquette agent (vx, vy)
    #     [9]     Angle raquette agent
    #     [10-11] Position adversaire (x, y)
    #     [12]    Balle de notre côté ? (1 = oui, -1 = non)
    #     [13]    Balle vient vers nous ? (1 = oui, -1 = non)
    #     [14]    Rebonds sur notre côté (0, 0.5, 1)
    #     [15]    Rebonds côté adverse (0, 0.5, 1)
    #     [16]    Distance balle-raquette normalisée
    #     [17]    Est-ce un service ? (1 = oui, -1 = non)
    #     """
    #     obs = np.zeros(18, dtype=np.float32)
        
    #     # Variables utiles
    #     agent_is_left = (self.agent_side == "left")
    #     # Offset adaptatif: +15 si balle va à droite, -15 si elle va à gauche
    #     velocity_offset = ADAPTIVE_BOUNDARY_OFFSET if (self.ball_in_play and self.ball and self.ball.vel[0] > 0) else (-ADAPTIVE_BOUNDARY_OFFSET if (self.ball_in_play and self.ball and self.ball.vel[0] < 0) else 0)
    #     net_center = WIDTH // 2 + velocity_offset
    #     paddle_center_x = self.agent_paddle.pos[0] + self.agent_paddle.width / 2
    #     paddle_center_y = self.agent_paddle.pos[1] + self.agent_paddle.height / 2
        
    #     if self.ball_in_play and self.ball is not None:
    #         ball_x = self.ball.pos[0]
    #         ball_y = self.ball.pos[1]
            
    #         # Position balle normalisée [0, 1] -> [-1, 1]
    #         obs[0] = (ball_x / WIDTH) * 2 - 1
    #         obs[1] = (ball_y / HEIGHT) * 2 - 1
            
    #         # Vitesse balle normalisée (max ~1000 px/s)
    #         max_vel = 1000.0
    #         obs[2] = np.clip(self.ball.vel[0] / max_vel, -1, 1)
    #         obs[3] = np.clip(self.ball.vel[1] / max_vel, -1, 1)
            
    #         # Spin normalisé (max ~500)
    #         max_spin = 500.0
    #         obs[4] = np.clip(self.ball.angular_speed / max_spin, -1, 1)
            
    #         # === NOUVELLES VARIABLES ===
            
    #         # Balle de notre côté ? (avec offset adaptatif basé sur la vélocité)
    #         velocity_offset = ADAPTIVE_BOUNDARY_OFFSET if self.ball.vel[0] > 0 else -ADAPTIVE_BOUNDARY_OFFSET
    #         net_center_offset = WIDTH // 2 + velocity_offset
    #         ball_on_agent_side = (ball_x < net_center_offset) if agent_is_left else (ball_x >= net_center_offset)
    #         obs[12] = 1.0 if ball_on_agent_side else -1.0
            
    #         # Balle vient vers nous ?
    #         ball_coming = (self.ball.vel[0] < 0) if agent_is_left else (self.ball.vel[0] > 0)
    #         obs[13] = 1.0 if ball_coming else -1.0
            
    #         # Rebonds sur notre côté (0, 0.5 pour 1, 1.0 pour 2+)
    #         our_bounces = self.ball.bounces_left if agent_is_left else self.ball.bounces_right
    #         obs[14] = min(our_bounces * 0.5, 1.0)
            
    #         # Rebonds côté adverse
    #         their_bounces = self.ball.bounces_right if agent_is_left else self.ball.bounces_left
    #         obs[15] = min(their_bounces * 0.5, 1.0)
            
    #         # Distance balle-raquette normalisée (max ~WIDTH)
    #         distance = np.sqrt((paddle_center_x - ball_x)**2 + (paddle_center_y - ball_y)**2)
    #         obs[16] = np.clip(distance / WIDTH, 0, 1) * 2 - 1  # [-1, 1]
            
    #         # Est-ce un service ?
    #         obs[17] = 1.0 if self.ball.service is not None else -1.0
        
    #     # Position raquette agent normalisée
    #     obs[5] = (self.agent_paddle.pos[0] / WIDTH) * 2 - 1
    #     obs[6] = (self.agent_paddle.pos[1] / HEIGHT) * 2 - 1
        
    #     # Vitesse raquette agent normalisée
    #     max_paddle_vel = 500.0
    #     obs[7] = np.clip(self.agent_paddle.vel[0] / max_paddle_vel, -1, 1)
    #     obs[8] = np.clip(self.agent_paddle.vel[1] / max_paddle_vel, -1, 1)
        
    #     # Angle raquette normalisé [-180, 180] -> [-1, 1]
    #     obs[9] = self.agent_paddle.angle / 180.0
        
    #     # Position adversaire normalisée
    #     obs[10] = (self.opponent_paddle.pos[0] / WIDTH) * 2 - 1
    #     obs[11] = (self.opponent_paddle.pos[1] / HEIGHT) * 2 - 1
        
    #     return obs

    def _get_winner_flag(self):
        """Retourne 'agent', 'opponent' ou None selon le vainqueur courant."""
        if self.point_winner_side is None:
            return None
        if (self.point_winner_side == 'left' and self.agent_side == 'left') or \
           (self.point_winner_side == 'right' and self.agent_side == 'right'):
            return "agent"
        return "opponent"
    
    def _is_ball_on_agent_side(self):
        """Retourne True si la balle est du côté de l'agent (avec offset adaptatif)."""
        if not self.ball_in_play or self.ball is None:
            return False
        # Offset adaptatif: +ADAPTIVE_BOUNDARY_OFFSET si balle va à droite, -ADAPTIVE_BOUNDARY_OFFSET si elle va à gauche
        velocity_offset = ADAPTIVE_BOUNDARY_OFFSET if self.ball.vel[0] > 0 else -ADAPTIVE_BOUNDARY_OFFSET
        net_center = WIDTH // 2 + velocity_offset
        agent_is_left = (self.agent_side == "left")
        ball_x = self.ball.pos[0]
        return (ball_x < net_center) if agent_is_left else (ball_x >= net_center)
    
    def _get_agent_side_bounces(self):
        """Retourne le nombre de rebonds du côté de l'agent."""
        if not self.ball_in_play or self.ball is None:
            return 0
        agent_is_left = (self.agent_side == "left")
        return self.ball.bounces_left if agent_is_left else self.ball.bounces_right
    
    def _get_opponent_side_bounces(self):
        """Retourne le nombre de rebonds du côté de l'adversaire."""
        if not self.ball_in_play or self.ball is None:
            return 0
        agent_is_left = (self.agent_side == "left")
        return self.ball.bounces_right if agent_is_left else self.ball.bounces_left
    
    def _update_scores(self, message=""):
        """Mise à jour des scores quand un point est marqué."""
        if self.point_winner_side == 'left':
            self.score_left += 1
        elif self.point_winner_side == 'right':
            self.score_right += 1
        
        self.last_point_message = message
    
    def render(self):
        """Affiche le jeu avec Pygame."""
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("Ping-Pong RL Training")
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
        
        # Consommer les événements Pygame pour éviter les bugs de clic
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        
        # Import des fonctions de rendu
        from graphics.renderer import (
            draw_background, draw_table, draw_ball, 
            draw_paddle, draw_net
        )
        
        draw_background(self.screen)
        draw_table(self.screen, self.table)
        
        if self.ball_in_play and self.ball is not None:
            draw_ball(self.screen, self.ball)
        
        # Agent en rouge, adversaire en noir
        draw_paddle(self.screen, self.agent_paddle, (255, 0, 0))
        draw_paddle(self.screen, self.opponent_paddle, (0, 0, 0))
        draw_net(self.screen, self.net)
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def close(self):
        """Ferme l'environnement."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
