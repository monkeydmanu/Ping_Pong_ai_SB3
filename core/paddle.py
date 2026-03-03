#si j'appuis sur un bouton alors je peux prendre le contrôle de la raquette de mon choix

"""
Classe représentant une raquette
"""

import numpy as np
from config import (
    HEIGHT,
    WIDTH,
    RACKET_HEIGHT_PX,
    RACKET_WIDTH_PX,
    SPEED_RACKET,
    FPS,
    PADDLE_SPEED_SCALE,
    PADDLE_ROT_SCALE,
)

class Paddle:
    def __init__(self, x, y, width=RACKET_WIDTH_PX, height=RACKET_HEIGHT_PX, speed=SPEED_RACKET, max_speed=None, x_min=0, x_max=WIDTH):
        self.pos = np.array([x, y], dtype=float)  # Position
        self.width = width
        self.height = height

        effective_speed = speed * PADDLE_SPEED_SCALE
        self.speed = effective_speed  # Vitesse maximale de déplacement (scalée)
        self.max_speed = (max_speed * PADDLE_SPEED_SCALE) if max_speed else effective_speed
        self.vel = np.array([0.0, 0.0], dtype=float)  # vélocité [vx, vy]
        self.angle = 0  # Rotation libre de la raquette
        self.acceleration = effective_speed * 30  # Accélération rapide, cohérente avec la vitesse scalée
        self.friction = 15  # Friction pour ralentir quand on lâche
        self.x_min = x_min  # Limite gauche
        self.x_max = x_max  # Limite droite
        self.can_hit = True  # Booléen pour savoir si la raquette peut toucher la balle
        
        # Historique des vitesses pour moyenne lissée (éviter faux contacts)
        self.velocity_history = []  # Liste des 5 dernières vitesses
        self.smoothed_vel = np.array([0.0, 0.0], dtype=float)  # Vitesse lissée
        
        # Vitesse angulaire (rotation du poignet)
        self.prev_angle = self.angle
        self.angular_velocity = 0.0  # Vitesse de rotation instantanée (°/s)
        self.angular_velocity_history = []  # Historique des 10 dernières vitesses angulaires
        self.smoothed_angular_velocity = 0.0  # Vitesse angulaire lissée

        # Debug collision: normale physique figée et durée d'affichage
        self.debug_contact_normal = np.array([0.0, 0.0], dtype=float)
        self.debug_contact_normal_steps = 0

    # Mise à jour de la position selon la vélocité et le dt
    def update(self, dt, speed_factor=1.0):
        """Met à jour la position de la raquette.
        
        Args:
            dt: Temps écoulé
            speed_factor: Facteur de curriculum (0.5 = lent, 1.0 = normal)
        """
        # Limiter la vitesse au max
        speed_magnitude = np.linalg.norm(self.vel)
        if speed_magnitude > self.max_speed:
            self.vel = self.vel / speed_magnitude * self.max_speed
        
        # Mettre à jour l'historique des vitesses (garder les 10 dernières frames)
        self.velocity_history.append(self.vel.copy())
        if len(self.velocity_history) > 10:
            self.velocity_history.pop(0)
        
        # Calculer la moyenne lissée (moyenne mobile des dernières vitesses)
        if len(self.velocity_history) > 0:
            self.smoothed_vel = np.mean(self.velocity_history, axis=0)
        else:
            self.smoothed_vel = self.vel.copy()
        
        # Calculer la vitesse angulaire (rotation du poignet)
        angle_diff = self.angle - self.prev_angle
        # Normaliser la différence d'angle dans [-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        self.angular_velocity = angle_diff / dt if dt > 0 else 0.0
        self.prev_angle = self.angle
        
        # Historique de la vitesse angulaire (garder les 10 dernières frames)
        self.angular_velocity_history.append(self.angular_velocity)
        if len(self.angular_velocity_history) > 10:
            self.angular_velocity_history.pop(0)
        
        # Moyenne lissée de la vitesse angulaire
        if len(self.angular_velocity_history) > 0:
            self.smoothed_angular_velocity = np.mean(self.angular_velocity_history)
        else:
            self.smoothed_angular_velocity = self.angular_velocity
        
        # Appliquer speed_factor à la vélocité pour curriculum
        self.pos += self.vel * dt * speed_factor

        # Limite verticale pour ne pas sortir de l'écran
        if self.pos[1] < 0:
            self.pos[1] = 0
            self.vel[1] = 0
        if self.pos[1] + self.height > HEIGHT:
            self.pos[1] = HEIGHT - self.height
            self.vel[1] = 0

        # Limite horizontale avec x_min et x_max
        if self.pos[0] < self.x_min:
            self.pos[0] = self.x_min
            self.vel[0] = 0
        if self.pos[0] + self.width > self.x_max:
            self.pos[0] = self.x_max - self.width
            self.vel[0] = 0

    # Mouvement vertical avec accélération
    def move_up(self, dt=None):
        if dt is None:
            dt = 1.0 / FPS
        self.vel[1] -= self.acceleration * dt  # accélère vers le haut

    def move_down(self, dt=None):
        if dt is None:
            dt = 1.0 / FPS
        self.vel[1] += self.acceleration * dt  # accélère vers le bas

    def stop_vertical(self):
        self.vel[1] = 0  # Arrêt instantané

    # Mouvement horizontal avec accélération
    def move_left(self, dt=None):
        if dt is None:
            dt = 1.0 / FPS
        self.vel[0] -= self.acceleration * dt

    def move_right(self, dt=None):
        if dt is None:
            dt = 1.0 / FPS
        self.vel[0] += self.acceleration * dt

    def stop_horizontal(self):
        self.vel[0] = 0  # Arrêt instantané

    # Rotation, sens trigo
    def rotate_left(self, dt, rotation_speed=6):
        self.angle -= (rotation_speed * PADDLE_ROT_SCALE) * dt
        self.angle %= 360

    def rotate_right(self, dt, rotation_speed=6):
        self.angle += (rotation_speed * PADDLE_ROT_SCALE) * dt
        self.angle %= 360

    # Retourne les infos essentielles pour collision/affichage
    def get_info(self):
        center_x = self.pos[0] + self.width / 2
        center_y = self.pos[1] + self.height / 2
        return center_x, center_y, self.vel[0], self.vel[1], self.angle, self.width, self.height
    
    def get_rect(self):
        return (self.pos[0], self.pos[1], self.width, self.height)