"""
Fonctions de collision balle ↔ raquette / filet / table
"""

import pygame
import numpy as np
from config import RESTITUTION, TABLE_Y, VPX_FRAME_MAX, WIDTH, FPS

# a = 0.35 pour de la mousse et 0.22 pour la table
# v0 = 200 m/s pour la mousse et 250 pour la table
def restitution_verticale(vy, v0, a, ey_min=0.85):
    """
    Calcule ey (coeff. de restitution vertical) en fonction de la vitesse verticale vy.

    Params:
      vy      : float, vitesse verticale (m/s) — signe ignoré (on utilise |vy|)
      v0      : float, vitesse de référence (m/s) — pour laquelle ey doit atteindre ey_min
      a       : float, exposant contrôlant la pente (plus a grand → moins de restitution)
      ey_min  : float, valeur minimale de ey (par défaut 0.85)

    Retour :
      ey : float dans [ey_min, 1]
    """
    # normalisation et clamp
    x = abs(vy) / v0
    if x >= 1.0:
        return float(ey_min)

    # noyau : (1 - x)^a décroît quand x augmente ; a plus grand → valeur plus petite
    ey = ey_min + (1.0 - ey_min) * (1.0 - x) ** a

    # sécurité numérique
    if ey < ey_min:
        ey = ey_min
    if ey > 1.0:
        ey = 1.0
    return float(ey)


# pour le moment on prend la même que y
# a = 0.35 pour de la mousse et 0.22 pour la table
# v0 = 200 m/s pour la mousse et 250 pour la table
def restitution_tangentielle(vx, v0, a, ex_min=0.85):
    """
    Calcule le coefficient de restitution tangentielle ex selon la formule empirique :
        ex = 1 - (vx / vx0)**a

    Paramètres :
    ------------
    vx : float
        Vitesse tangentielle (horizontale) avant impact (m/s)
    vx0 : float
        Vitesse caractéristique liée à la friction (m/s)
    a : float
        Exposant empirique lié au type de surface

    Retour :
    --------
    ex : float
        Coefficient de restitution tangentielle (sans unité)
    """

    # normalisation et clamp
    z = abs(vx) / v0
    if z >= 1.0:
        return float(ex_min)

    # noyau : (1 - x)^a décroît quand x augmente ; a plus grand → valeur plus petite
    ex = ex_min + (1.0 - ex_min) * (1.0 - z) ** a

    # sécurité numérique
    if ex < ex_min:
        ex = ex_min
    if ex > 1.0:
        ex = 1.0
    return ex


def adjust_spin_for_corner(angular_speed, ratio, is_left_corner=True):
    """
    Ajuste la vitesse angulaire selon le coin et le ratio.
    - Si la spin est déjà dans le bon sens → augmente avec ratio
    - Sinon → tend vers le bon signe progressivement
    """
    # Coin gauche : bon sens = négatif
    if is_left_corner:
        if angular_speed < 0:
            return angular_speed * (1 + 0.5 * ratio)  # augmente
        else:
            return angular_speed * (1 - 0.8 * ratio) - 0.8 * ratio * abs(angular_speed)  # tend vers négatif
    # Coin droit : bon sens = positif
    else:
        if angular_speed > 0:
            return angular_speed * (1 + 0.5 * ratio)  # augmente
        else:
            return angular_speed * (1 - 0.8 * ratio) + 0.8 * ratio * abs(angular_speed)  # tend vers positif

def reduction_speed(vx, vy, est_mousse):
    if est_mousse:
        vx *= restitution_tangentielle(vx, VPX_FRAME_MAX, 0.35)
        vy *= restitution_verticale(vy, VPX_FRAME_MAX, 0.35)
    else:
        vx *= restitution_tangentielle(vx, VPX_FRAME_MAX*1.2, 0.22)
        vy *= restitution_verticale(vy, VPX_FRAME_MAX*1.2, 0.22)
    return vx, vy


def contact_cercle_rectangle(ball_center_x, ball_center_y, radius,
                             rect_x, rect_y, width, height, angle_deg, screen=None, est_mousse=False):
    """
    Système : 0° = Haut, Rotation Horaire.
    Repère : Pygame (Y+ vers le bas).
    """

    # 0) Setup
    half_w = width / 2.0
    half_h = height / 2.0
    rect_cx = rect_x + half_w
    rect_cy = rect_y + half_h

    # 1) Angle en radians
    theta = np.radians(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # 2) Translation
    dx = ball_center_x - rect_cx
    dy = ball_center_y - rect_cy

    # 3) Rotation inverse (pour passer du monde au repère local du rectangle)
    # Matrice de rotation horaire inverse : 
    # rx =  dx * cos + dy * sin
    # ry = -dx * sin + dy * cos
    rx = dx * cos_t + dy * sin_t
    ry = -dx * sin_t + dy * cos_t

    # 4) Trouver le point le plus proche sur le rectangle local (aligné sur les axes)
    closest_x = np.clip(rx, -half_w, half_w)
    closest_y = np.clip(ry, -half_h, half_h)

    # 5) Test de collision
    dist_x = rx - closest_x
    dist_y = ry - closest_y
    distance_sq = dist_x*dist_x + dist_y*dist_y
    hit = distance_sq <= radius*radius

    # Initialisation variables retour
    contact_local = (closest_x, closest_y)
    face = None
    corner_ratio = None
    n_local = (0.0, 0.0)

    # 6) & 7) Classification de la face et normale locale
    outside_x = (rx < -half_w) or (rx > half_w)
    outside_y = (ry < -half_h) or (ry > half_h)

    if outside_x and not outside_y:
        face = 'gauche' if rx < -half_w else 'droite'
        n_local = (-1.0, 0.0) if rx < -half_w else (1.0, 0.0)
    elif outside_y and not outside_x:
        face = 'haut' if ry < -half_h else 'bas'
        n_local = (0.0, -1.0) if ry < -half_h else (0.0, 1.0)
    elif (not outside_x) and (not outside_y):
        # Cas où le centre est à l'intérieur
        dist_to_right = half_w - rx
        dist_to_left  = rx + half_w
        dist_to_top   = ry + half_h
        dist_to_bottom= half_h - ry
        min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        if min_dist == dist_to_left:   face = 'gauche'; n_local = (-1.0, 0.0)
        elif min_dist == dist_to_right: face = 'droite'; n_local = (1.0, 0.0)
        elif min_dist == dist_to_top:   face = 'haut';   n_local = (0.0, -1.0)
        else:                           face = 'bas';    n_local = (0.0, 1.0)
    else:
        # Coins
        left = rx < -half_w
        top = ry < -half_h
        face = 'corner_' + ('h' if top else 'b') + ('g' if left else 'd')
        # Ratio pour l'effet
        dist_edge_x = abs(rx) - half_w
        corner_ratio = min(1.0, abs(dist_edge_x) / radius)
        
        nx_local = rx - closest_x
        ny_local = ry - closest_y
        length = np.hypot(nx_local, ny_local)
        n_local = (nx_local/length, ny_local/length) if length != 0 else (0,0)

    # 8) Projection du point de contact : Local -> Monde
    # Matrice de rotation horaire : 
    # wx =  rx * cos - ry * sin
    # wy =  rx * sin + ry * cos
    contact_world_x = closest_x * cos_t - closest_y * sin_t + rect_cx
    contact_world_y = closest_x * sin_t + closest_y * cos_t + rect_cy
    contact_world = (contact_world_x, contact_world_y)

    # 9) Projection de la normale : Local -> Monde
    # On utilise LA MÊME matrice que l'étape 8
    nx_world = n_local[0] * cos_t - n_local[1] * sin_t
    ny_world = n_local[0] * sin_t + n_local[1] * cos_t
    
    # Normalisation de sécurité
    norm = np.hypot(nx_world, ny_world)
    if norm != 0:
        nx_world /= norm
        ny_world /= norm
    normal_world = (nx_world, ny_world)

    # 10) Tangente
    # Pour un rebond, la tangente est souvent utile pour le spin
    tangent_world = (-ny_world, nx_world)

    # 11) Dessin debug
    if screen is not None:
        corners_local = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        corners_world = []
        for cx, cy in corners_local:
            wx = cx * cos_t + cy * sin_t + rect_cx
            wy = -cx * sin_t + cy * cos_t + rect_cy
            corners_world.append((int(wx), int(wy)))
        pygame.draw.polygon(screen, (0, 255, 0), corners_world, 2)
        if hit:
            pygame.draw.circle(screen, (0, 0, 255), (int(contact_world[0]), int(contact_world[1])), 4)
            # Dessin de la normale en rouge
            pygame.draw.line(screen, (255, 0, 0), contact_world, 
                             (contact_world[0] + nx_world * 20, contact_world[1] + ny_world * 20), 2)

    return hit, contact_world, normal_world, tangent_world, face, corner_ratio


# a=0.35 pour la mousse et 0.22 pour la table
def check_rect_collision(ball, rectangle, est_mousse, est_table, a, spin_factor=0.2, screen=None):
    """
    Gestion du rebond sur un rectangle avec physique de vitesse relative.
    """
    ball_center_x = ball.pos[0]
    ball_center_y = ball.pos[1]

    # Récupération des infos du rectangle
    if est_mousse:
        _, _, vel_x, vel_y, angle, width, height = rectangle.get_info()
        # Lissage vitesse
        if hasattr(rectangle, 'smoothed_vel'):
            vel_x, vel_y = rectangle.smoothed_vel[0], rectangle.smoothed_vel[1]
        x, y, _, _ = rectangle.get_rect()
    elif est_table:
        x, y, width, height = rectangle.get_rect()
        vel_x, vel_y = 0, 0
        angle = 0 
    else: # filet
        x, y, width, height = rectangle.get_rect()
        vel_x, vel_y = 0, 0
        angle = 0 

    # Détection collision géométrique
    hit, contact, normal, tangent, face, corner_ratio = contact_cercle_rectangle(
        ball_center_x, ball_center_y, ball.radius,
        x, y, width, height, angle, screen, est_mousse
    )

    if not hit:
        return

    # # Debug info
    # if est_mousse:
    #     print(f"Collision détectée: Face={face}")
    #     print(f"  Raquette V=({vel_x:.1f}, {vel_y:.1f})")
    #     print(f"  Balle V=({ball.vel[0]:.1f}, {ball.vel[1]:.1f})")

    # Repositionnement immédiat pour éviter que la balle reste collée
    ball.pos[0] = contact[0] + normal[0] * ball.radius
    ball.pos[1] = contact[1] + normal[1] * ball.radius

    # --- 1. Calcul de la Vitesse Relative ---
    # On se place dans le référentiel de la raquette
    rel_vx = ball.vel[0] - vel_x
    rel_vy = ball.vel[1] - vel_y

    # --- 2. Vérification de la direction d'impact ---
    # Produit scalaire : V_rel . Normale
    v_dot_n = rel_vx * normal[0] + rel_vy * normal[1]

    # Si v_dot_n > 0, la balle s'éloigne déjà de la surface (ou la raquette la fuit)
    # On ne fait rien, sauf si on veut éviter le "tunneling"
    if v_dot_n >= 0:
        return

    # --- Gestion Spéciale des Coins (Gardée de ton code) ---
    if face and face.startswith('corner_'):
        # Logique simplifiée pour les coins : on inverse juste selon la normale du coin + effet
        # (Tu peux réintégrer ta logique complexe ici si tu veux, mais attention aux signes)
        ball.vel[0] -= 2 * v_dot_n * normal[0]
        ball.vel[1] -= 2 * v_dot_n * normal[1]
        # On ajoute un peu de chaos/spin comme avant
        return

    # --- 3. Réflexion Physique (Rebond) ---
    # Formule : V_new = V_old - (1 + restitution) * (V_old . N) * N
    # On applique un coefficient de restitution (rebond)
    # Pour le ping pong : 
    #   restitution verticale ~ 0.8 (table) à 1.2 (mousse active qui pousse)
    #   restitution horizontale dépend du spin
    
    coeff_restitution = 1.0  # Base élastique
    if est_mousse:
        # La mousse "pousse" un peu (effet trampoline des gommes modernes)
        coeff_restitution = 0.85
    elif est_table:
        coeff_restitution = 0.85

    # Calcul du vecteur de rebond en vitesse relative
    # j = impulsion scalaire
    j = -(1 + coeff_restitution) * v_dot_n
    
    rel_vx_new = rel_vx + j * normal[0]
    rel_vy_new = rel_vy + j * normal[1]

    # --- 4. Friction / Tangente (Effet) ---
    # Vitesse tangentielle relative
    v_dot_t = rel_vx * tangent[0] + rel_vy * tangent[1]
    
    # Friction de la surface (ralentit la balle tangentiellement)
    friction = 0.8 if est_mousse else 0.95
    rel_vx_new -= (1 - friction) * v_dot_t * tangent[0]
    rel_vy_new -= (1 - friction) * v_dot_t * tangent[1]

    # Ajout du "coup de poignet" (Spin generation)
    if est_mousse:
        # Si la raquette frotte la balle, on ajoute du spin
        # Vitesse tangentielle relative détermine le spin généré
        spin_generation = v_dot_t * 0.5  # Facteur arbitraire
        ball.angular_speed -= spin_generation 
        
        # Et inversement, le spin existant modifie la trajectoire (Effet Magnus au rebond)
        grip = 0.3 # Adhérence
        tangential_kick = ball.angular_speed * grip * 0.1
        rel_vx_new += tangential_kick * tangent[0]
        rel_vy_new += tangential_kick * tangent[1]

    # --- 5. Retour au Monde ---
    ball.vel[0] = rel_vx_new + vel_x
    ball.vel[1] = rel_vy_new + vel_y
    
    # Limites globales (clamp)
    ball.vel[0], ball.vel[1] = reduction_speed(ball.vel[0], ball.vel[1], est_mousse)

    # if est_mousse:
    #     print(f"  [Rebond Relatif] v_dot_n={v_dot_n:.1f}")
    #     print(f"  [Resultat] Balle V=({ball.vel[0]:.1f}, {ball.vel[1]:.1f})")





def check_table_collision(ball, table):
    """Vérifie la collision avec la table et track les rebonds."""
    from config import WIDTH
    
    # Sauvegarder la vitesse Y avant collision
    old_vel_y = ball.vel[1]
    old_pos_y = ball.pos[1]
    
    check_rect_collision(ball, table, est_mousse=False, est_table=True, a=0.22)
    
    # Si la balle allait vers le bas et maintenant va vers le haut = rebond
    if old_vel_y > 0 and ball.vel[1] < 0:
        net_center = WIDTH // 2
        if ball.pos[0] < net_center:
            ball.bounces_left += 1
        else:
            ball.bounces_right += 1
        return True
    
    return False



def check_ball_paddle(ball, paddle, screen):
    # Vérifier si la raquette peut toucher la balle
    if not paddle.can_hit:
        return
    
    # Sauvegarder la position avant collision pour détecter si collision a eu lieu
    old_pos = ball.pos.copy()
    
    check_rect_collision(ball, paddle, est_mousse=True, est_table=False, a=0.35, screen=screen)

    
    # Si la position a changé, une collision a eu lieu
    if not np.array_equal(old_pos, ball.pos):
        paddle.can_hit = False  # La raquette ne peut plus toucher la balle


def check_ball_net(ball, net, restitution=RESTITUTION, spin_factor=0.3, spin_damping=0.8):
    """
    Gestion de la collision avec le filet :
    - ball : instance de Ball
    - net : instance de Net
    - restitution : coefficient pour vel_x inversé
    - spin_factor : influence de l'angular_speed sur vel_y
    - spin_damping : perte de spin après collision
    """
    check_rect_collision(ball, net, est_mousse=False, est_table=False, a=0.22)


