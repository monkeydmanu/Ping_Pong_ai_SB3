"""
Classe représentant la balle et sa physique.
"""

import random
import numpy as np
from config import GRAVITY, BALL_RADIUS, FPS, BALL_SPEED_SCALE
from engine.collision import check_table_collision

class Ball:
    def __init__(self, x, y, vx, vy, radius=BALL_RADIUS, angular_speed=0):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([vx, vy], dtype=float)
        self.radius = radius
        self.angle = 90      # angle actuel pour l'affichage
        self.angular_speed = angular_speed      # rad/s
        
        # Flags physiques
        self.gravity_enabled = True  # peut être désactivée pour un spawn statique
        
        # Tracking des rebonds sur la table
        self.bounces_left = 0   # Nombre de rebonds sur la moitié gauche
        self.bounces_right = 0  # Nombre de rebonds sur la moitié droite
        self.last_hit_by = None  # 'left' ou 'right' - qui a frappé en dernier
        self.service = None  # None, 'left', ou 'right' - côté qui serve, None quand service est terminé

    def update(self, dt=None, speed_factor=1.0):
        """Met à jour la position et la vitesse de la balle avec effet Magnus et traînée.
        
        Args:
            dt: Temps écoulé (1/FPS si None)
            speed_factor: Facteur de curriculum (0.5 = balle lente, 1.0 = normale)
        """
        if dt is None:
            # Utiliser le FPS unique pour la physique
            dt = 1.0 / FPS

        # Échelle globale pour ralentir ou accélérer la physique de la balle
        # Appliquer speed_factor ici pour curriculum learning
        dt_scaled = dt * BALL_SPEED_SCALE * speed_factor
        
        # Vitesse en pixels/s pour calculer la norme
        speed_px = np.linalg.norm(self.vel)
        
        if speed_px > 1.0:  # éviter division par zéro
            # === EFFET MAGNUS (simplifié pour le jeu) ===
            magnus_strength = 0.5  # ajustable pour plus/moins d'effet
            
            # Topspin (angular_speed > 0 avec vel[0] > 0) → force vers le bas (+y)
            if self.vel[0] != 0:
                magnus_accel_y = magnus_strength * self.angular_speed * np.sign(self.vel[0])
                self.vel[1] += magnus_accel_y * dt_scaled
            
            # Composante horizontale (plus faible)
            if self.vel[1] != 0:
                magnus_accel_x = -magnus_strength * 0.3 * self.angular_speed * np.sign(self.vel[1])
                self.vel[0] += magnus_accel_x * dt_scaled
            
            # === TRAÎNÉE AÉRODYNAMIQUE (légère) ===
            drag_factor = 0.5  # par seconde
            self.vel[0] *= (1 - drag_factor * dt_scaled)
            self.vel[1] *= (1 - drag_factor * dt_scaled)
        
        # Gravité : GRAVITY=9.81 m/s², converti en pixels/s² (si activée)
        if self.gravity_enabled:
            gravity_pixels = GRAVITY * 200  # ~2000 pixels/s²
            self.vel[1] += gravity_pixels * dt_scaled
        
        # Translation
        self.pos += self.vel * dt_scaled
        
        # Rotation visuelle
        self.angle += self.angular_speed * dt_scaled
        
        # Décroissance naturelle du spin
        self.angular_speed *= (1 - 0.1 * dt_scaled)  # ~10% par seconde


def _spawn_ball_static(table, side, game_mode=False):
    """Crée une balle immobile pour un service (vx=0)."""
    x_table, y_table, w_table, h_table = table.get_rect()

    x_offset = 70 if game_mode else random.randint(50, 200)
    y_offset = 300

    x = x_table + x_offset if side == "left" else x_table + w_table - x_offset
    y = y_table - y_offset

    return Ball(
        x=x,
        y=y,
        vx=0,
        vy=0,
        angular_speed=0
    )


def spawn_ball_left(table, game_mode=False, train_phase=0):
    """Crée une balle au bord gauche de la table (service gauche)."""
    ball = _spawn_ball_static(table, "left", game_mode=game_mode)
    ball.service = 'left'  # Service depuis la gauche
    return ball


def spawn_ball_right(table, game_mode=False, train_phase=0):
    """Crée une balle au bord droit de la table (service droite)."""
    ball = _spawn_ball_static(table, "right", game_mode=game_mode)

    if game_mode: # pour que ball.service = None en mode entrainement car sinon service_faut tjs True
        ball.service = 'right'  # Service depuis la droite

    return ball


def spawn_ball_right_to_left(table):
    """Crée une balle côté droit avec vitesse vers la gauche (entrainement)."""
    x_table, y_table, w_table, h_table = table.get_rect()

    return Ball(
        x=x_table + w_table - 200,
        y=y_table - 250,
        vx=random.randint(-600, -300),
        vy=random.randint(-300, -250),
        angular_speed=random.randint(-100, 100)
    )

def spawn_ball_right_to_left_defense(table):
    """Crée une balle côté droit avec vitesse vers la gauche (entrainement)."""
    x_table, y_table, w_table, h_table = table.get_rect()

    prob = random.randint(1, 2) # soit 1 soit 2 pour faire varier les trajectoires entre rapide et lent

    if prob == 1:
        return Ball(
            x=x_table + w_table - 300,
            y=y_table - 100,
            vx= random.randint(-800, -500), #random.randint(-800, -500)
            vy= random.randint(-100, 1500), #random.randint(-100, 1500)
            angular_speed=random.randint(-300, 300) #random.randint(-300, 300)
        )
    else: # balle lente pour amortir
        return Ball(
            x=x_table + w_table - 250,
            y=y_table - 100,
            vx= random.randint(-400, -200), #random.randint(-400, -200)
            vy= random.randint(-1000, -200), #random.randint(-1000, -200)
            angular_speed= random.randint(-300, 300) #random.randint(-300, 300)
        )


def spawn_ball_left_to_right(table):
    """Crée une balle côté gauche avec vitesse vers la droite (entrainement)."""
    x_table, y_table, w_table, h_table = table.get_rect()

    return Ball(
        x=x_table + 200,
        y=y_table - 250,
        vx=random.randint(300, 600),
        vy=random.randint(-300, -250),
        angular_speed=random.randint(-100, 100)
    )

def spawn_ball_left_to_right_defense(table):
    """Crée une balle côté gauche avec vitesse vers la droite (entrainement)."""
    x_table, y_table, w_table, h_table = table.get_rect()

    prob = random.randint(1, 2) # soit 1 soit 2 pour faire varier les trajectoires entre rapide et lent
    
    if prob == 1:
        return Ball(
            x=x_table + 300,
            y=y_table - 100,
            vx= random.randint(500, 800), #random.randint(-800, -500)
            vy= random.randint(-100, 1500), #random.randint(-100, 1500)
            angular_speed=random.randint(-300, 300) #random.randint(-300, 300)
        )
    else: # balle lente pour amortir
        return Ball(
            x=x_table + 250,
            y=y_table - 100,
            vx= random.randint(200, 400), #random.randint(-400, -200)
            vy= random.randint(-1000, -200), #random.randint(-1000, -200)
            angular_speed= random.randint(-300, 300) #random.randint(-300, 300)
        )